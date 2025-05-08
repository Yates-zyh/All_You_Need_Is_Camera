import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import librosa
from tqdm import tqdm
import glob
import joblib
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
FRAME_RATE = 100
DATASET_PATH = "./ImperialDance/ImperialDance"
GENRES = ['b', 'k', 'j', 'h', 'u']
EXPERTISE_LEVELS = {'b': 0, 'i': 1, 'e': 2}

# Joint mapping
JOINT_MAPPING = {
    5: 5,    # RShoulder
    6: 9,    # LShoulder
    7: 17,   # RUArm
    8: 18,   # LUArm
    9: 20,   # RFArm
    10: 8,   # LFArm
    11: 19,  # Hip
    12: 2,   # RThigh
    13: 3,   # LThigh
    14: 0,   # RShin
    15: 12,  # LShin
    16: 10,  # RFoot
}

class DanceFeatureExtractor:
    def __init__(self):
        self.max_seq_len = 1000
    
    def _extract_motion_features(self, motion_data):
        """提取运动特征，包含加速度和平滑度"""
        try:
            # 处理输入维度
            if len(motion_data.shape) == 4:  # [samples, coords, frames, joints]
                motion_data = motion_data[0]  # 取第一个样本
            elif len(motion_data.shape) == 3:  # [coords, frames, joints]
                motion_data = motion_data
            else:
                raise ValueError(f"Unexpected motion data shape: {motion_data.shape}")
            
            # 确保是2D坐标
            if motion_data.shape[0] > 2:
                motion_data = motion_data[:2]
            
            coords_dim, total_frames, num_joints = motion_data.shape
            
            # 截断或填充到max_seq_len
            if total_frames > self.max_seq_len:
                start = (total_frames - self.max_seq_len) // 2
                motion_data = motion_data[:, start:start + self.max_seq_len, :]
                total_frames = self.max_seq_len
            elif total_frames < self.max_seq_len:
                padding_size = self.max_seq_len - total_frames
                motion_data = np.pad(
                    motion_data,
                    ((0, 0), (0, padding_size), (0, 0)),
                    mode='constant',
                    constant_values=0
                )
            
            # 计算身体尺度用于归一化
            if JOINT_MAPPING[11] < num_joints and JOINT_MAPPING[5] < num_joints:  # Hip and RShoulder
                hip_pos = motion_data[:, :, JOINT_MAPPING[11]]
                shoulder_pos = motion_data[:, :, JOINT_MAPPING[5]]
                body_scale = np.mean(np.sqrt(np.sum((hip_pos - shoulder_pos)**2, axis=0)))
                body_scale = max(body_scale, 1e-6)  # 避免除零
            else:
                body_scale = 1.0
            
            # 1. 计算关键关节对之间的距离
            joint_pairs = [
                (JOINT_MAPPING[5], JOINT_MAPPING[7]),   # RShoulder-RUArm
                (JOINT_MAPPING[6], JOINT_MAPPING[8]),   # LShoulder-LUArm
                (JOINT_MAPPING[7], JOINT_MAPPING[9]),   # RUArm-RFArm
                (JOINT_MAPPING[8], JOINT_MAPPING[10]),  # LUArm-LFArm
                (JOINT_MAPPING[11], JOINT_MAPPING[12]), # Hip-RThigh
                (JOINT_MAPPING[11], JOINT_MAPPING[13]), # Hip-LThigh
            ]
            
            distances = np.zeros((total_frames, len(joint_pairs)))
            for t in range(total_frames):
                for i, (joint1, joint2) in enumerate(joint_pairs):
                    if joint1 < num_joints and joint2 < num_joints:
                        dist = np.sqrt(np.sum((motion_data[:, t, joint1] - motion_data[:, t, joint2])**2))
                        distances[t, i] = dist / body_scale
            
            # 2. 计算速度和加速度
            velocities = np.zeros((total_frames-1, len(joint_pairs)))
            accelerations = np.zeros((total_frames-2, len(joint_pairs)))
            
            for i in range(len(joint_pairs)):
                # 计算速度
                velocities[:, i] = np.abs(np.diff(distances[:, i]))
                # 计算加速度
                accelerations[:, i] = np.abs(np.diff(velocities[:, i]))
            
            # 3. 计算运动平滑度
            smoothness = np.zeros(len(joint_pairs))
            for i in range(len(joint_pairs)):
                # 使用速度的方差作为平滑度指标
                smoothness[i] = np.var(velocities[:, i])
            
            # 4. 计算统计特征
            features = []
            
            # 距离特征
            mean_distances = np.mean(distances, axis=0)
            std_distances = np.std(distances, axis=0)
            max_distances = np.max(distances, axis=0)
            
            # 速度特征
            mean_velocity = np.mean(velocities, axis=0)
            std_velocity = np.std(velocities, axis=0)
            max_velocity = np.max(velocities, axis=0)
            
            # 加速度特征
            mean_acceleration = np.mean(accelerations, axis=0)
            std_acceleration = np.std(accelerations, axis=0)
            max_acceleration = np.max(accelerations, axis=0)
            
            # 平滑度特征
            mean_smoothness = np.mean(smoothness)
            std_smoothness = np.std(smoothness)
            max_smoothness = np.max(smoothness)
            
            # 组合所有特征
            features.extend(mean_distances)      # 6个特征
            features.extend(std_distances)       # 6个特征
            features.extend(max_distances)       # 6个特征
            features.extend(mean_velocity)       # 6个特征
            features.extend(std_velocity)        # 6个特征
            features.extend(max_velocity)        # 6个特征
            features.extend(mean_acceleration)   # 6个特征
            features.extend(std_acceleration)    # 6个特征
            features.extend(max_acceleration)    # 6个特征
            features.extend([mean_smoothness, std_smoothness, max_smoothness])  # 3个特征
            
            # 确保特征维度为51
            features = np.array(features, dtype=np.float32)
            if len(features) > 51:
                features = features[:51]
            elif len(features) < 51:
                features = np.pad(features, (0, 51 - len(features)), mode='constant', constant_values=0.0)
            
            return features
            
        except Exception as e:
            print(f"Error in motion feature extraction: {e}")
            return np.zeros(51, dtype=np.float32)
    
    def _extract_audio_features(self, audio_path):
        """提取音频特征"""
        try:
            # 统一采样率
            y, sr = librosa.load(audio_path, sr=22050)
            
            # 1. 节拍特征
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            
            # 节拍强度
            beat_strength = float(np.mean(onset_env))
            beat_strength_std = float(np.std(onset_env))
            
            # 节拍间隔稳定性
            beat_intervals = np.diff(beat_times)
            beat_stability = float(1.0 / (np.std(beat_intervals) + 1e-6))
            
            # 2. 节奏特征
            # 计算节奏变化
            rhythm_change = float(np.std(np.diff(beat_frames)))
            
            # 计算节奏复杂度
            rhythm_complexity = float(np.mean(np.abs(np.diff(onset_env))))
            
            # 3. 能量特征
            # 计算RMS能量
            rms = librosa.feature.rms(y=y)[0]
            energy_mean = float(np.mean(rms))
            energy_std = float(np.std(rms))
            
            # 计算能量变化率
            energy_change = float(np.mean(np.abs(np.diff(rms))))
            
            # 4. 频谱特征
            # 计算频谱质心
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            centroid_mean = float(np.mean(spectral_centroid))
            centroid_std = float(np.std(spectral_centroid))
            
            # 计算频谱带宽
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            bandwidth_mean = float(np.mean(spectral_bandwidth))
            bandwidth_std = float(np.std(spectral_bandwidth))
            
            # 组合所有特征
            features = [
                # 节拍特征
                beat_strength, beat_strength_std, beat_stability,
                # 节奏特征
                rhythm_change, rhythm_complexity,
                # 能量特征
                energy_mean, energy_std, energy_change,
                # 频谱特征
                centroid_mean, centroid_std,
                bandwidth_mean, bandwidth_std
            ]
            
            # 确保特征维度为12
            features = np.array(features, dtype=np.float32)
            if len(features) > 12:
                features = features[:12]
            elif len(features) < 12:
                features = np.pad(features, (0, 12 - len(features)), mode='constant', constant_values=0.0)
            
            return features
            
        except Exception as e:
            print(f"Error extracting audio features: {e}")
            return np.zeros(12, dtype=np.float32)

def collect_dance_files():
    """Collect all paths to dance files"""
    print("Collecting dance files...")
    data_tuples = []
    
    for genre in GENRES:
        for chapter_dir in glob.glob(f"{DATASET_PATH}/{genre}_ch*"):
            base_name = os.path.basename(chapter_dir)
            audio_path = f"{chapter_dir}/{base_name}_audio.mp3"
            
            for level in EXPERTISE_LEVELS.keys():
                motion_path = f"{chapter_dir}/dance_level_{level}_{base_name}.npy"
                
                if os.path.exists(motion_path) and os.path.exists(audio_path):
                    expertise_level = EXPERTISE_LEVELS[level]
                    data_tuples.append((motion_path, audio_path, expertise_level))
    
    print(f"Collected {len(data_tuples)} dance files")
    return data_tuples

def extract_features(data_tuples, feature_extractor):
    """提取所有样本的特征"""
    print("Extracting features...")
    X = []
    y = []
    
    for motion_path, audio_path, expertise_level in tqdm(data_tuples):
        # 加载动作数据
        motion_data = np.load(motion_path)
        
        # 确保motion_data是4D的 [samples, coords, frames, joints]
        if len(motion_data.shape) == 3:
            motion_data = motion_data[np.newaxis, ...]
        
        # 处理每个段
        for segment_idx in range(min(100, motion_data.shape[0])):
            # 获取当前段的数据
            segment_data = motion_data[segment_idx]
            
            # 提取特征
            motion_features = feature_extractor._extract_motion_features(segment_data)
            audio_features = feature_extractor._extract_audio_features(audio_path)
            
            # 组合特征
            features = np.concatenate([motion_features, audio_features])
            X.append(features)
            y.append(expertise_level)
    
    return np.array(X), np.array(y)

def train_and_evaluate(X_train, X_val, y_train, y_val):
    """训练和评估随机森林模型"""
    print("Training Random Forest model...")
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 创建随机森林分类器
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # 训练模型
    rf.fit(X_train_scaled, y_train)
    
    # 评估模型
    y_pred = rf.predict(X_val_scaled)
    accuracy = accuracy_score(y_val, y_pred)
    print("\nValidation Accuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    # 保存模型和标准化器
    joblib.dump(rf, 'random_forest_model.joblib')
    joblib.dump(scaler, 'feature_scaler.joblib')
    
    # 特征重要性可视化
    feature_importance = rf.feature_importances_
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.title('Feature Importance')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.savefig('feature_importance.png')
    plt.close()
    
    return rf, scaler

def main():
    # 收集数据
    data_tuples = collect_dance_files()
    
    # 创建特征提取器
    feature_extractor = DanceFeatureExtractor()
    
    # 提取特征
    X, y = extract_features(data_tuples, feature_extractor)
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(X))
    indices = np.random.permutation(len(X))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]
    
    # 训练和评估模型
    model, scaler = train_and_evaluate(X_train, X_val, y_train, y_val)
    
    print("\nTraining complete.")
    print("Model saved as 'random_forest_model.joblib'")
    print("Feature scaler saved as 'feature_scaler.joblib'")
    print("Feature importance plot saved as 'feature_importance.png'")

if __name__ == "__main__":
    main() 