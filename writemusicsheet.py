"""
Dance Rhythm Game Music Sheet Generator
舞蹈节奏游戏谱面生成器

This script analyzes dance videos and generates rhythm game music sheets (charts).
该脚本分析舞蹈视频并生成节奏游戏谱面文件。

How to use 使用方法:
1. Run the script directly: python writemusicsheet.py
   直接运行脚本：python writemusicsheet.py
2. Enter the path to a dance video when prompted
   在提示时输入舞蹈视频的路径
3. The script will automatically:
   脚本将自动：
   - Extract audio from the video 从视频中提取音频
   - Detect beats in the audio 检测音频中的节拍
   - Extract dancer's pose at each beat point 在每个节拍点提取舞者姿势
   - Generate music sheets for different difficulty levels 生成不同难度的谱面

Output 输出:
- Location 位置: ./musicsheet/<video_name>/
- Format 格式: Three JSON files will be generated for each video:
              每个视频将生成三个JSON文件：
  * <video_name>_easy.json - 4 keypoints (Left/Right Wrist, Left/Right Ankle)
                             4个关键点（左/右手腕，左/右脚踝）
  * <video_name>_normal.json - 8 keypoints (Easy + Left/Right Hip, Left/Right Shoulder)
                              8个关键点（简单难度的点 + 左/右髋部，左/右肩部）
  * <video_name>_hard.json - 13 keypoints (All except eyes and ears)
                            13个关键点（除了眼睛和耳朵外的所有关键点）

Requirements 依赖:
- Python 3.6+
- OpenCV
- NumPy
- PyTorch
- YOLO11x-pose.pt model
- librosa
- ffmpeg (must be in PATH)

Note: GPU acceleration is used if available.
注意：如果可用，将使用GPU加速。
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import torch
import time
import subprocess
import tempfile
import json
from ultralytics import YOLO
import librosa
import tqdm

class MusicSheetGenerator:
    def __init__(self, video_path, beat_sample_rate=1.0, hop_length=512):
        print("Initializing Dance Beat Pose Extraction System...")
        
        # Initialize YOLO11Pose model
        print("Loading YOLO model...")
        self.model = YOLO('yolo11x-pose.pt')
        print("YOLO model loaded successfully")
        
        # Target video
        print(f"Opening video: {video_path}")
        self.video = cv2.VideoCapture(video_path)
        if not self.video.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        self.video_path = video_path
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps
        print(f"Video info: {self.width}x{self.height}, {self.fps}fps, {self.duration:.2f} seconds")
        
        # Extract audio file
        self.audio_path = self.extract_audio()
        
        # Beat detection parameters
        self.beat_sample_rate = beat_sample_rate  # 节拍采样率：1.0表示使用所有检测到的节拍，0.5表示每两个节拍取一个
        self.hop_length = hop_length  # librosa参数，影响节拍检测的精度，较小值会检测到更多节拍
        
        # Beat detection results
        self.beat_frames = []
        self.tempo = 0
        
        # Keypoint extraction parameters
        self.confidence_threshold = 0.2
        
        # Define keypoint names
        self.keypoint_names = [
            "Nose", "Right Eye", "Left Eye", "Right Ear", "Left Ear", 
            "Right Shoulder", "Left Shoulder", "Right Elbow", "Left Elbow", "Right Wrist", "Left Wrist",
            "Right Hip", "Left Hip", "Right Knee", "Left Knee", "Right Ankle", "Left Ankle"
        ]
        
        # Beat pose storage
        self.beat_poses = []
        
        print("Initialization complete")
    
    def extract_audio(self):
        """Extract audio from video"""
        try:
            print("Extracting audio...")
            
            # Create audio file in current directory instead of temp dir
            video_name = os.path.splitext(os.path.basename(self.video_path))[0]
            audio_path = f"{video_name}_audio.wav"  # Use wav for better librosa compatibility
            
            # Use ffmpeg to extract audio
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", self.video_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "44100", "-ac", "2",
                audio_path
            ]
            
            print(f"Executing command: {' '.join(ffmpeg_cmd)}")
            
            # Create process
            process = subprocess.Popen(
                ffmpeg_cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Set up progress bar
            pbar = tqdm.tqdm(total=100, desc="Extracting audio")
            
            stderr_output = []
            # Read output in real-time and update progress bar
            while process.poll() is None:
                line = process.stderr.readline()
                stderr_output.append(line)
                if 'time=' in line:
                    try:
                        time_str = line.split('time=')[1].split()[0]
                        h, m, s = time_str.split(':')
                        current_time = float(h) * 3600 + float(m) * 60 + float(s)
                        progress = min(int(current_time / self.duration * 100), 100)
                        pbar.update(progress - pbar.n)
                    except:
                        pass
                time.sleep(0.1)
            
            # Get remaining output
            stdout, stderr = process.communicate()
            stderr_output.append(stderr if stderr else "")
            pbar.close()
            
            # Check for success
            if process.returncode != 0:
                print("Audio extraction failed, error details:")
                print("\n".join(stderr_output))
                return None
            
            # Check generated file
            if not os.path.exists(audio_path):
                print(f"Error: Output file {audio_path} does not exist")
                return None
                
            if os.path.getsize(audio_path) == 0:
                print(f"Error: Output file {audio_path} is empty")
                return None
                
            print(f"Audio extracted to: {audio_path}")
            return audio_path
                
        except Exception as e:
            print(f"Error while extracting audio: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def detect_beats(self):
        """Detect beats in the audio file"""
        if not self.audio_path:
            print("No audio file available")
            return []
        
        print("Detecting beats from audio...")
        try:
            # Load audio file
            y, sr = librosa.load(self.audio_path)
            
            # Extract tempo and beat frames
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length)
            # Store tempo as a float, not a numpy array
            self.tempo = float(tempo)
            
            # Convert beat frames to timestamps
            beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=self.hop_length)
            
            # Convert beat times to video frame numbers
            frame_numbers = np.round(beat_times * self.fps).astype(int)
            
            # Apply beat sample rate if different from default
            if self.beat_sample_rate != 1.0:
                # Calculate sampling interval based on beat_sample_rate
                sample_interval = max(1, int(1/self.beat_sample_rate))
                sampled_frame_numbers = frame_numbers[::sample_interval]
                print(f"Applied beat sampling rate {self.beat_sample_rate} (taking 1 beat every {sample_interval})")
            else:
                sampled_frame_numbers = frame_numbers
            
            # Convert to Python list to avoid NumPy array boolean issues
            self.beat_frames = sampled_frame_numbers.tolist()
            
            print(f"Detected tempo: {self.tempo:.2f} BPM")
            print(f"Found {len(frame_numbers)} total beats, using {len(self.beat_frames)} after sampling")
            
            return self.beat_frames
            
        except Exception as e:
            print(f"Error detecting beats with librosa: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def extract_keypoints(self, keypoints_obj):
        """Extract numpy array from YOLO Keypoints object"""
        try:
            if keypoints_obj is None:
                return None
                
            # First move CUDA tensor to CPU, then convert to numpy
            kp_data = keypoints_obj.data.cpu().numpy()
            
            # Ensure we only get the first person's keypoints (remove batch dimension)
            kp_data = kp_data.squeeze(0)
            
            return kp_data
            
        except Exception as e:
            print(f"Error extracting keypoints: {e}")
            return None
    
    def analyze_pose_at_beat(self, frame_idx):
        """Analyze pose at the specified frame (beat)"""
        # Set video position to beat frame
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # Read frame
        ret, frame = self.video.read()
        if not ret:
            print(f"Could not read frame {frame_idx}")
            return None
        
        # Use YOLO model to detect pose
        results = self.model(frame, verbose=False)
        
        # Extract keypoints
        keypoints_raw = None
        if len(results) > 0 and hasattr(results[0], 'keypoints') and len(results[0].keypoints) > 0:
            keypoints_raw = results[0].keypoints[0]
        
        keypoints = self.extract_keypoints(keypoints_raw)
        
        # Calculate time position (seconds)
        time_pos = frame_idx / self.fps
        
        # Return pose data (without storing the frame to save memory)
        pose_data = {
            'frame_idx': frame_idx,
            'time': time_pos,
            'keypoints': keypoints.tolist() if keypoints is not None else None
        }
        
        return pose_data
    
    def extract_poses_at_beats(self):
        """Extract poses at all detected beat frames"""
        if len(self.beat_frames) == 0:
            print("No beats detected. Running beat detection first...")
            self.detect_beats()
            
        if len(self.beat_frames) == 0:
            print("Beat detection failed")
            return []
        
        print("Extracting poses at beat points...")
        beat_poses = []
        
        # Filter beat frames to ensure they're within video range
        valid_beats = [b for b in self.beat_frames if 0 <= b < self.frame_count]
        
        # Process each beat frame
        pbar = tqdm.tqdm(total=len(valid_beats), desc="Processing beat frames")
        for i, beat_frame in enumerate(valid_beats):
            pose = self.analyze_pose_at_beat(beat_frame)
            if pose is not None:
                beat_poses.append(pose)
            pbar.update(1)
        
        pbar.close()
        
        print(f"Extracted {len(beat_poses)} beat-synchronized poses")
        self.beat_poses = beat_poses
        
        return beat_poses
    
    def generate_music_sheet(self, output_path="music_sheet"):
        """Generate a music sheet JSON file"""
        if len(self.beat_poses) == 0:
            print("No beat poses found. Running extraction first...")
            self.extract_poses_at_beats()
            
        if len(self.beat_poses) == 0:
            print("No beat poses extracted")
            return False
        
        # Get base output filename (without extension)
        if output_path.endswith(".json"):
            base_name = output_path[:-5]  # Remove .json
        else:
            base_name = output_path
        
        # Create output directory
        video_name = os.path.basename(base_name)
        output_dir = os.path.join("musicsheet", video_name)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Creating output directory: {output_dir}")
        
        # Define keypoint indices for different difficulty levels
        easy_keypoints = [9, 10, 15, 16]  # Left/Right Wrist, Left/Right Ankle
        medium_keypoints = [9, 10, 15, 16, 11, 12, 5, 6]  # Easy + Left/Right Hip, Left/Right Shoulder
        hard_keypoints = list(range(17))  # All keypoints except [1, 2, 3, 4] (eyes and ears)
        hard_keypoints = [i for i in hard_keypoints if i not in [1, 2, 3, 4]]
        
        # Keypoint names for reference (full set)
        keypoint_names_full = self.keypoint_names.copy()
        
        # Generate music sheets for different difficulty levels
        success = True
        results = []
        
        # Define difficulty levels and their corresponding keypoints
        difficulty_levels = {
            "easy": easy_keypoints,
            "normal": medium_keypoints,
            "hard": hard_keypoints
        }
        
        # Process each difficulty level
        for difficulty, keypoint_indices in difficulty_levels.items():
            # Create selected keypoint names
            selected_keypoint_names = [keypoint_names_full[i] for i in keypoint_indices]
            
            # Create music sheet data structure
            music_sheet = {
                'difficulty': difficulty,
                'video_info': {
                    'width': self.width,
                    'height': self.height,
                    'fps': self.fps,
                    'duration': self.duration,
                    'frame_count': self.frame_count
                },
                'audio_info': {
                    'tempo': self.tempo
                },
                'keypoint_indices': keypoint_indices,
                'keypoint_names': selected_keypoint_names,
                'beats': []
            }
            
            # Add each beat's data with selected keypoints
            for pose in self.beat_poses:
                # Create beat data with only selected keypoints
                if pose['keypoints'] is not None:
                    # Extract only the selected keypoints
                    selected_keypoints = [pose['keypoints'][i] if i < len(pose['keypoints']) else None for i in keypoint_indices]
                    
                    # Remove confidence values (keep only x, y coordinates)
                    if selected_keypoints:
                        selected_keypoints = [[point[0], point[1]] if point is not None else None for point in selected_keypoints]
                else:
                    selected_keypoints = None
                
                beat_data = {
                    'frame_idx': pose['frame_idx'],
                    'time': pose['time'],
                    'keypoints': selected_keypoints
                }
                
                music_sheet['beats'].append(beat_data)
            
            # Generate output path for this difficulty
            difficulty_output_path = os.path.join(output_dir, f"{video_name}_{difficulty}.json")
            
            # Write to JSON file
            with open(difficulty_output_path, 'w') as f:
                json.dump(music_sheet, f, indent=2)
            
            print(f"Generated {difficulty} music sheet with {len(music_sheet['beats'])} beats at {difficulty_output_path}")
            results.append(difficulty_output_path)
        
        return success
    
    def cleanup(self):
        """Clean up resources"""
        print("\nCleaning up resources...")
        self.video.release()
        
        # Delete audio file
        if self.audio_path and os.path.exists(self.audio_path):
            try:
                os.remove(self.audio_path)
                print(f"Deleted audio file: {self.audio_path}")
            except Exception as e:
                print(f"Could not delete audio file: {e}")
                
        print("Resource cleanup complete")
    
    def run(self, output_path="music_sheet"):
        """Run the complete music sheet generation process"""
        try:
            # Detect beats
            self.detect_beats()
            
            # Extract poses at beat points
            self.extract_poses_at_beats()
            
            # Generate music sheet
            success = self.generate_music_sheet(output_path)
            
            # Clean up resources
            self.cleanup()
            
            if success:
                print(f"Music sheet generation complete!")
                return True
            else:
                print("Failed to generate music sheet")
                return False
                
        except Exception as e:
            print(f"Error during music sheet generation: {e}")
            import traceback
            traceback.print_exc()
            self.cleanup()
            return False


def main():
    """Main function to run the program"""
    print("=== Music Sheet Generator ===")
    
    # Print system information
    print(f"Python version: {os.sys.version}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Get video path from user
    video_path = input("Enter dance video path: ")
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} does not exist")
        return
    
    # Get beat sampling rate
    while True:
        beat_sample_rate_str = input("Enter beat sampling rate (default 1.0, lower values = fewer beats, e.g. 0.5): ")
        if beat_sample_rate_str == "":
            beat_sample_rate = 1.0
            break
        try:
            beat_sample_rate = float(beat_sample_rate_str)
            if beat_sample_rate <= 0:
                print("Error: Beat sampling rate must be positive")
            else:
                break
        except ValueError:
            print("Error: Please enter a valid number")
    
    # Get hop_length for librosa
    while True:
        hop_length_str = input("Enter hop_length for beat detection (default 512, lower values detect more beats): ")
        if hop_length_str == "":
            hop_length = 512
            break
        try:
            hop_length = int(hop_length_str)
            if hop_length <= 0:
                print("Error: hop_length must be positive")
            else:
                break
        except ValueError:
            print("Error: Please enter a valid integer")
    
    # Extract video filename without extension
    video_basename = os.path.basename(video_path)
    video_name = os.path.splitext(video_basename)[0]
    
    # Automatically generate output path
    output_path = video_name  # No .json extension - will be added per difficulty
    print(f"Output will be saved in: musicsheet/{video_name}/")
    
    # Create and run generator
    generator = MusicSheetGenerator(video_path, beat_sample_rate, hop_length)
    result = generator.run(output_path)
    
    if result:
        print(f"\nMusic sheet JSON files created successfully!")
        print(f"Files are located in: musicsheet/{video_name}/")
        print("You can use these files for rhythm game chart generation.")
    else:
        print("\nFailed to create music sheet JSON files.")


if __name__ == "__main__":
    main()