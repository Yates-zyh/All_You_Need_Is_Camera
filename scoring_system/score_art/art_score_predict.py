import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import joblib
from tqdm import tqdm
import librosa
from sklearn.preprocessing import StandardScaler
from train_art_score_model import DanceFeatureExtractor, JOINT_MAPPING
from collections import Counter

# Load pre-trained YOLOv11-pose model
model = YOLO('yolo11x-pose.pt')

def extract_audio_features(audio_data, sr):
    """Extract audio features from audio data"""
    try:
        # 1. Beat features
        onset_env = librosa.onset.onset_strength(y=audio_data, sr=sr)
        tempo, beat_frames = librosa.beat.beat_track(y=audio_data, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        
        # Beat strength
        beat_strength = float(np.mean(onset_env))
        beat_strength_std = float(np.std(onset_env))
        
        # Beat stability
        beat_intervals = np.diff(beat_times)
        beat_stability = float(1.0 / (np.std(beat_intervals) + 1e-6))
        
        # 2. Rhythm features
        rhythm_change = float(np.std(np.diff(beat_frames)))
        rhythm_complexity = float(np.mean(np.abs(np.diff(onset_env))))
        
        # 3. Energy features
        rms = librosa.feature.rms(y=audio_data)[0]
        energy_mean = float(np.mean(rms))
        energy_std = float(np.std(rms))
        energy_change = float(np.mean(np.abs(np.diff(rms))))
        
        # 4. Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
        centroid_mean = float(np.mean(spectral_centroid))
        centroid_std = float(np.std(spectral_centroid))
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)[0]
        bandwidth_mean = float(np.mean(spectral_bandwidth))
        bandwidth_std = float(np.std(spectral_bandwidth))
        
        # Combine all features
        features = [
            # Beat features
            beat_strength, beat_strength_std, beat_stability,
            # Rhythm features
            rhythm_change, rhythm_complexity,
            # Energy features
            energy_mean, energy_std, energy_change,
            # Spectral features
            centroid_mean, centroid_std,
            bandwidth_mean, bandwidth_std
        ]
        
        # Convert to numpy array
        features = np.array(features, dtype=np.float32)
        
        # Ensure we have exactly 12 features
        assert len(features) == 12, f"Expected 12 audio features, got {len(features)}"
        
        return features
        
    except Exception as e:
        print(f"Error extracting audio features: {e}")
        return np.zeros(12, dtype=np.float32)

def extract_motion_features(motion_data):
    """Extract motion features from pose data"""
    try:
        # 1. Calculate extension (using key joint pairs)
        key_joint_pairs = [
            (0, 1),   # Left-Right Shoulder
            (2, 3),   # Left-Right Elbow
            (4, 5),   # Left-Right Wrist
            (6, 7),   # Left-Right Hip
            (8, 9),   # Left-Right Knee
            (10, 11)  # Left-Right Ankle
        ]
        
        # Initialize arrays for distances, velocities, and accelerations
        distances = np.zeros((motion_data.shape[1], len(key_joint_pairs)))
        velocities = np.zeros((motion_data.shape[1]-1, len(key_joint_pairs)))
        accelerations = np.zeros((motion_data.shape[1]-2, len(key_joint_pairs)))
        
        # Calculate distances for each frame
        for t in range(motion_data.shape[1]):
            for i, (joint1, joint2) in enumerate(key_joint_pairs):
                dist = np.sqrt(np.sum((motion_data[:, t, joint1] - motion_data[:, t, joint2])**2))
                distances[t, i] = dist
        
        # Calculate velocities
        for i in range(len(key_joint_pairs)):
            velocities[:, i] = np.abs(np.diff(distances[:, i]))
        
        # Calculate accelerations
        for i in range(len(key_joint_pairs)):
            accelerations[:, i] = np.abs(np.diff(velocities[:, i]))
        
        # Calculate smoothness (variance of velocities)
        smoothness = np.zeros(len(key_joint_pairs))
        for i in range(len(key_joint_pairs)):
            smoothness[i] = np.var(velocities[:, i])
        
        # Calculate statistical features
        features = []
        
        # Distance features (18 features)
        mean_distances = np.mean(distances, axis=0)
        std_distances = np.std(distances, axis=0)
        max_distances = np.max(distances, axis=0)
        features.extend(mean_distances)
        features.extend(std_distances)
        features.extend(max_distances)
        
        # Velocity features (18 features)
        mean_velocity = np.mean(velocities, axis=0)
        std_velocity = np.std(velocities, axis=0)
        max_velocity = np.max(velocities, axis=0)
        features.extend(mean_velocity)
        features.extend(std_velocity)
        features.extend(max_velocity)
        
        # Acceleration features (12 features)
        mean_acceleration = np.mean(accelerations, axis=0)
        std_acceleration = np.std(accelerations, axis=0)
        features.extend(mean_acceleration)
        features.extend(std_acceleration)
        
        # Smoothness features (3 features)
        mean_smoothness = np.mean(smoothness)
        std_smoothness = np.std(smoothness)
        max_smoothness = np.max(smoothness)
        features.extend([mean_smoothness, std_smoothness, max_smoothness])
        
        # Convert to numpy array
        features = np.array(features, dtype=np.float32)
        
        # Ensure we have exactly 51 features
        assert len(features) == 51, f"Expected 51 motion features, got {len(features)}"
        
        return features
        
    except Exception as e:
        print(f"Error in motion feature extraction: {e}")
        return np.zeros(51, dtype=np.float32)

def process_segment(video_frames, audio_segment, fps, sr):
    """Process a 15-second segment of video and audio"""
    try:
        # Process video frames
        all_poses = []
        for frame in video_frames:
            # Use YOLOv11-pose for prediction
            results = model(frame)
            
            # Extract keypoints
            if len(results) > 0 and len(results[0].keypoints) > 0:
                # Get keypoint data and ensure consistent shape
                keypoints = results[0].keypoints[0].data[0].cpu().numpy()
                
                # Only keep x, y coordinates
                if len(keypoints.shape) == 2:
                    keypoints_2d = keypoints[:, :2]  # Only take x,y coordinates
                    all_poses.append(keypoints_2d)
                else:
                    print(f"Unexpected keypoint shape: {keypoints.shape}")
                    continue
        
        if not all_poses:
            print("No poses detected in the segment")
            return None
        
        # Ensure all pose data has consistent shape
        max_joints = max(pose.shape[0] for pose in all_poses)
        standardized_poses = []
        
        for pose in all_poses:
            # If joint count is insufficient, pad with zeros
            if pose.shape[0] < max_joints:
                padded_pose = np.zeros((max_joints, 2))  # Changed to 2D
                padded_pose[:pose.shape[0]] = pose
                standardized_poses.append(padded_pose)
            else:
                standardized_poses.append(pose)
        
        # Convert to numpy array
        all_poses = np.array(standardized_poses)
        
        # Convert pose data to model required format
        motion_data = np.zeros((2, len(all_poses), 21))  # Changed to 2D coordinates
        
        # Map keypoints to model required format
        for i, pose in enumerate(all_poses):
            for src_joint, dst_joint in JOINT_MAPPING.items():
                if src_joint < pose.shape[0]:
                    motion_data[:, i, dst_joint] = pose[src_joint][:2]  # Only use x,y coordinates
        
        # Extract motion features
        motion_features = extract_motion_features(motion_data)
        
        # Extract audio features
        audio_features = extract_audio_features(audio_segment, sr)
        
        # Combine all features
        combined_features = np.concatenate([
            motion_features,  # 51 features
            audio_features    # 12 features
        ])
        
        # Ensure we have exactly 63 features
        assert len(combined_features) == 63, f"Expected 63 features, got {len(combined_features)}"
        
        return combined_features
        
    except Exception as e:
        print(f"Error processing segment: {e}")
        return None

def process_whole_video(video_path):
    """处理整个视频并提取特征"""
    try:
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return None
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 存储所有帧的姿态数据
        all_poses = []
        
        print("Processing video frames...")
        for _ in tqdm(range(total_frames)):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Use YOLOv11-pose for prediction
            results = model(frame)
            
            # Extract keypoints
            if len(results) > 0 and len(results[0].keypoints) > 0:
                # Get keypoint data and ensure consistent shape
                keypoints = results[0].keypoints[0].data[0].cpu().numpy()
                
                # Only keep x, y coordinates
                if len(keypoints.shape) == 2:
                    keypoints_2d = keypoints[:, :2]  # Only take x,y coordinates
                    all_poses.append(keypoints_2d)
                else:
                    print(f"Unexpected keypoint shape: {keypoints.shape}")
                    continue
        
        cap.release()
        
        if not all_poses:
            print("No poses detected in the video")
            return None
        
        # Ensure all pose data has consistent shape
        max_joints = max(pose.shape[0] for pose in all_poses)
        standardized_poses = []
        
        for pose in all_poses:
            # If joint count is insufficient, pad with zeros
            if pose.shape[0] < max_joints:
                padded_pose = np.zeros((max_joints, 2))  # Changed to 2D
                padded_pose[:pose.shape[0]] = pose
                standardized_poses.append(padded_pose)
            else:
                standardized_poses.append(pose)
        
        # Convert to numpy array
        all_poses = np.array(standardized_poses)
        
        # Convert pose data to model required format
        motion_data = np.zeros((2, len(all_poses), 21))  # Changed to 2D coordinates
        
        # Map keypoints to model required format
        for i, pose in enumerate(all_poses):
            for src_joint, dst_joint in JOINT_MAPPING.items():
                if src_joint < pose.shape[0]:
                    motion_data[:, i, dst_joint] = pose[src_joint][:2]  # Only use x,y coordinates
        
        # Extract motion features
        motion_features = extract_motion_features(motion_data)
        
        # Extract audio features
        y, sr = librosa.load(video_path, sr=22050)
        audio_features = extract_audio_features(y, sr)
        
        # Combine all features
        combined_features = np.concatenate([
            motion_features,  # 51 features
            audio_features    # 12 features
        ])
        
        # Ensure we have exactly 63 features
        assert len(combined_features) == 63, f"Expected 63 features, got {len(combined_features)}"
        
        return combined_features
        
    except Exception as e:
        print(f"Error processing whole video: {e}")
        return None

def predict_dance_level(video_path):
    """预测舞蹈水平，同时使用整体评估和分段评估"""
    try:
        # 加载模型和标准化器
        model = joblib.load('./Trained_models/random_forest_model.joblib')
        scaler = joblib.load('./Trained_models/feature_scaler.joblib')
        
        # 1. 整体评估
        print("Performing overall evaluation...")
        overall_features = process_whole_video(video_path)
        if overall_features is not None:
            # 标准化特征
            overall_features_scaled = scaler.transform(overall_features.reshape(1, -1))
            
            # 预测
            overall_prediction = model.predict(overall_features_scaled)[0]
            overall_probabilities = model.predict_proba(overall_features_scaled)[0]
        
        # 2. 分段评估
        print("\nPerforming segment-wise evaluation...")
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return None
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        # 加载音频
        y, sr = librosa.load(video_path, sr=22050)
        
        # 存储每个片段的预测结果
        segment_predictions = []
        
        # 处理每个15秒片段
        for start_time in np.arange(0, duration, 15):
            end_time = min(start_time + 15, duration)
            
            # 计算当前片段的帧范围
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            
            # 读取视频帧
            video_frames = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for _ in range(end_frame - start_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                video_frames.append(frame)
            
            # 提取对应的音频片段
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            audio_segment = y[start_sample:end_sample]
            
            # 处理当前片段
            features = process_segment(video_frames, audio_segment, fps, sr)
            if features is not None:
                # 标准化特征
                features_scaled = scaler.transform(features.reshape(1, -1))
                
                # 预测
                prediction = model.predict(features_scaled)[0]
                segment_predictions.append(prediction)
        
        cap.release()
        
        if not segment_predictions:
            print("No valid segment predictions made")
            return None
        
        # 获取预测的舞蹈水平
        level_map = {0: 'Beginner', 1: 'Intermediate', 2: 'Expert'}
        
        return {
            'overall': {
                'level': level_map[overall_prediction],
                'probabilities': {
                    'Beginner': overall_probabilities[0] * 100,
                    'Intermediate': overall_probabilities[1] * 100,
                    'Expert': overall_probabilities[2] * 100
                }
            },
            'segments': {
                'predictions': [level_map[p] for p in segment_predictions],
                'total_duration': duration
            }
        }
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

def main():
    # 设置视频文件路径
    video_path = "D:\VisionProject\earthquake.mp4"  # 替换为实际的视频文件路径
    
    # 预测舞蹈水平
    result = predict_dance_level(video_path)
    
    if result:
        print("\nOverall Evaluation:")
        # 获取概率最高的级别
        max_prob = max(result['overall']['probabilities'].items(), key=lambda x: x[1])
        print(f"Final Rating: {max_prob[0]} ({max_prob[1]:.2f}%)")
        print("\nProbabilities for each level:")
        for level, prob in result['overall']['probabilities'].items():
            print(f"{level}: {prob:.2f}%")
        
        print("\nSegment-wise predictions:")
        for i, pred in enumerate(result['segments']['predictions']):
            start_time = i * 15
            end_time = min((i + 1) * 15, result['segments']['total_duration'])
            print(f"{int(start_time//60):02d}:{int(start_time%60):02d} - {int(end_time//60):02d}:{int(end_time%60):02d}: {pred}")
    else:
        print("Prediction failed.")

if __name__ == "__main__":
    main() 