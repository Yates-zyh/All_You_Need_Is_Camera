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
  * <video_name>_easy.json - Same 4 keypoints (Right/Left Wrist, Right/Left Ankle) at 0.25 sampling rate (1 beat every 4 beats)
                             相同的4个关键点（右/左手腕，右/左脚踝），采样率0.25（每4拍取1拍）
  * <video_name>_normal.json - Same 4 keypoints at 0.5 sampling rate (1 beat every 2 beats)
                              相同的4个关键点，采样率0.5（每2拍取1拍）
  * <video_name>_hard.json - Same 4 keypoints at 1.0 sampling rate (every beat)
                            相同的4个关键点，采样率1.0（每拍都取）

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
        
        # Define keypoints to use for all difficulty levels (same 4 points)
        selected_keypoints = [9, 10, 15, 16]  # Right Wrist, Left Wrist, Right Ankle, Left Ankle
        
        # Define difficulty levels based on sampling frequency
        difficulty_levels = {
            "easy": 0.25,    # Every 4 beats
            "normal": 0.5,   # Every 2 beats
            "hard": 1.0      # Every beat
        }
        
        # Keypoint names for reference
        keypoint_names_full = self.keypoint_names.copy()
        selected_keypoint_names = [keypoint_names_full[i] for i in selected_keypoints]
        
        # Generate music sheets for different difficulty levels
        success = True
        results = []
        
        # Process each difficulty level
        for difficulty, sampling_rate in difficulty_levels.items():
            print(f"Generating {difficulty} difficulty with sampling rate {sampling_rate}")
            
            # Apply sampling rate to determine which beats to include
            if sampling_rate < 1.0:
                sample_interval = max(1, int(1/sampling_rate))
                sampled_beats = self.beat_poses[::sample_interval]
                print(f"  - Sampling 1 beat every {sample_interval} beats ({len(sampled_beats)} of {len(self.beat_poses)} total beats)")
            else:
                sampled_beats = self.beat_poses
                print(f"  - Using all {len(sampled_beats)} beats")
            
            # Create music sheet data structure
            music_sheet = {
                'difficulty': difficulty,
                'sampling_rate': sampling_rate,
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
                'keypoint_indices': selected_keypoints,
                'keypoint_names': selected_keypoint_names,
                'beats': []
            }
            
            # Add each beat's data with selected keypoints
            for pose in sampled_beats:
                # Create beat data with only selected keypoints
                if pose['keypoints'] is not None:
                    # Extract only the selected keypoints
                    selected_kp = [pose['keypoints'][i] if i < len(pose['keypoints']) else None for i in selected_keypoints]
                    
                    # Remove confidence values (keep only x, y coordinates)
                    if selected_kp:
                        selected_kp = [[point[0], point[1]] if point is not None else None for point in selected_kp]
                else:
                    selected_kp = None
                
                beat_data = {
                    'frame_idx': pose['frame_idx'],
                    'time': pose['time'],
                    'keypoints': selected_kp
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
    
    # Using default beat_sample_rate = 1.0 instead of asking user
    # This will be adjusted per difficulty level in generate_music_sheet
    beat_sample_rate = 1.0
    print("Using fixed sampling rates for different difficulty levels:")
    print("  - Easy: 0.25 (1 beat every 4 beats)")
    print("  - Normal: 0.5 (1 beat every 2 beats)")
    print("  - Hard: 1.0 (every beat)")
    
    # Using default hop_length without asking user
    hop_length = 512
    print(f"Using default hop_length: {hop_length} for beat detection")
    
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