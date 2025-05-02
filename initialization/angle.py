import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def calculate_3d_angles(landmarks, frame_shape):
    """计算三维空间三个轴向的倾斜角度"""
    h, w = frame_shape[:2]
    angles = {'X': 0, 'Y': 0, 'Z': 0}
    
    try:
        # 获取关键点坐标（标准化到0-1范围）
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        
        # X轴：侧向倾斜（左右倾斜）→ 比较双肩高度差
        shoulder_center = (left_shoulder.y + right_shoulder.y)/2
        hip_center = (left_hip.y + right_hip.y)/2
        vertical_diff = shoulder_center - hip_center
        angles['X'] = np.degrees(np.arcsin(vertical_diff*2))  # 扩大差异灵敏度
        
        # Y轴：前后倾倒 → 鼻子到踝关节的垂直投影差
        nose_projection = nose.y
        ankle_projection = left_ankle.y
        forward_lean = nose_projection - ankle_projection
        angles['Y'] = np.degrees(np.arcsin(forward_lean*2))  # 正值表示前倾
        
        # Z轴：身体旋转 → 双肩连线与双髋连线的夹角
        shoulder_vector = np.array([right_shoulder.x - left_shoulder.x, 
                                   right_shoulder.y - left_shoulder.y])
        hip_vector = np.array([right_hip.x - left_hip.x, 
                              right_hip.y - left_hip.y])
        cos_theta = np.dot(shoulder_vector, hip_vector)/(
            np.linalg.norm(shoulder_vector)*np.linalg.norm(hip_vector)+1e-7)
        angles['Z'] = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
        
    except:
        pass
    
    return angles

def draw_angle_indicator(frame, angles):
    """绘制三维角度指示器"""
    h, w = frame.shape[:2]
    
    # 在画面左上角绘制角度表
    cv2.putText(frame, "3D Posture Analysis", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # X轴（侧向倾斜）
    color_x = (0, 165, 255) if abs(angles['X']) > 10 else (0, 255, 0)
    cv2.putText(frame, f"X: Lateral Tilt {angles['X']:.1f}deg", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_x, 2)
    
    # Y轴（前后倾倒）
    color_y = (0, 165, 255) if abs(angles['Y']) > 15 else (0, 255, 0)
    cv2.putText(frame, f"Y: Forward Lean {angles['Y']:.1f}deg", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_y, 2)
    
    # Z轴（身体旋转）
    color_z = (0, 165, 255) if abs(angles['Z']) > 20 else (0, 255, 0)
    cv2.putText(frame, f"Z: Body Rotation {angles['Z']:.1f}deg", (10, 120),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_z, 2)
    
    # 在画面右侧绘制简易三维坐标系
    cv2.line(frame, (w-100, h//2), (w-50, h//2), (0, 0, 255), 2)  # X轴（红色）
    cv2.line(frame, (w-75, h//2-25), (w-75, h//2+25), (0, 255, 0), 2)  # Y轴（绿色）
    cv2.circle(frame, (w-75, h//2), 8, (255, 0, 0), -1)  # Z轴指示（蓝色圆点）
    
    return frame

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        output_frame = frame.copy()
        
        if results.pose_landmarks:
            angles = calculate_3d_angles(results.pose_landmarks.landmark, frame.shape)
            mp_drawing.draw_landmarks(output_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            output_frame = draw_angle_indicator(output_frame, angles)
        else:
            cv2.putText(output_frame, "No Body Detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('3D Posture Analysis', output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = '/Users/macbookair/Downloads/IMG_0505.MOV'
    process_video(video_path)
