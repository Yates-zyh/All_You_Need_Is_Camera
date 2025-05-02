import cv2
import numpy as np
import mediapipe as mp
from math import radians, cos, sin
from distance_pose_height import distance_detection
from angle import calculate_3d_angles, draw_angle_indicator

# Initialize MediaPipe Pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Rotation matrices
def get_rotation_matrix(pitch, yaw, roll):
    """Generate a 3D rotation matrix for pitch, yaw, and roll."""
    pitch = radians(pitch)
    yaw = radians(yaw)
    roll = radians(roll)

    # Rotation matrix for pitch (X-axis)
    Rx = np.array([[1, 0, 0],
                   [0, cos(pitch), -sin(pitch)],
                   [0, sin(pitch), cos(pitch)]])
    
    # Rotation matrix for yaw (Y-axis)
    Ry = np.array([[cos(yaw), 0, sin(yaw)],
                   [0, 1, 0],
                   [-sin(yaw), 0, cos(yaw)]])
    
    # Rotation matrix for roll (Z-axis)
    Rz = np.array([[cos(roll), -sin(roll), 0],
                   [sin(roll), cos(roll), 0],
                   [0, 0, 1]])

    # Combined rotation matrix
    rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))
    
    return rotation_matrix

def apply_rotation(landmarks, rotation_matrix, frame_shape):
    """Apply rotation to the landmarks and return transformed landmarks."""
    h, w, _ = frame_shape
    transformed_landmarks = []
    
    for landmark in landmarks:
        point = np.array([landmark.x * w, landmark.y * h, 0])
        
        # Check for NaN values in the landmarks
        if np.isnan(point[0]) or np.isnan(point[1]):
            continue  # Skip this landmark if it's invalid
        
        # Apply rotation
        rotated_point = np.dot(rotation_matrix, point)
        
        # Scale to original frame size

        transformed_landmarks.append(rotated_point)
    return transformed_landmarks

# Function to calculate pixel distance between two keypoints
def get_pixel_distance(keypoint1, keypoint2):
    return np.linalg.norm(np.array(keypoint1) - np.array(keypoint2))

# Normalize pose to standard plane with scaling (90% height) and center alignment
def get_scaling_factor(landmarks, frame_shape, target_height_ratio=0.9):
    h, w, _ = frame_shape
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

    # Calculate head to toe distance (body height in pixels)
    head_to_toe_distance = get_pixel_distance([nose.x * w, nose.y * h], 
                                              [(left_ankle.x + right_ankle.x) / 2 * w, 
                                               (left_ankle.y + right_ankle.y) / 2 * h])

    # Calculate the scaling factor for 90% of screen height
    target_distance = target_height_ratio * h
    scaling_factor = target_distance / head_to_toe_distance
    return scaling_factor

# Visualize original and scaled landmarks in a comparison window
def visualize_pose_comparison(frame, original_landmarks, scaled_landmarks, distance, angles):
    output_frame = frame.copy()

    # Draw original landmarks in red
    for lmk in original_landmarks:
        x = int(lmk.x * frame.shape[1])
        y = int(lmk.y * frame.shape[0])
        cv2.circle(output_frame, (x, y), 3, (0, 0, 255), -1)  # Red for original landmarks

    # Draw scaled landmarks in green
    for lmk in scaled_landmarks:
        x = int(lmk[0])  # Normalized x-coordinate
        y = int(lmk[1])  # Normalized y-coordinate
        cv2.circle(output_frame, (x, y), 3, (0, 255, 0), -1)  # Green for transformed landmarks

    return output_frame

# Process video frames
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  # Rotate the frame if needed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        processed_frame, distance, distance_text = distance_detection(frame_rgb, height=1.7, focal_length=600)

        if results.pose_landmarks:
            # Visualize original landmarks
            original_landmarks = results.pose_landmarks.landmark
            
            # Calculate 3D angles
            angles = calculate_3d_angles(results.pose_landmarks.landmark, frame.shape)
            mp_drawing.draw_landmarks(processed_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Get rotation matrix based on the calculated angles
            rotation_matrix = get_rotation_matrix(angles['X'], angles['Y'], angles['Z'])
            
            # Apply rotation to landmarks
            transformed_landmarks = apply_rotation(results.pose_landmarks.landmark, rotation_matrix, frame.shape)
            
            # Draw original landmarks in red
            for lmk in results.pose_landmarks.landmark:
                x = int(lmk.x * frame.shape[1])
                y = int(lmk.y * frame.shape[0])
                cv2.circle(processed_frame, (x, y), 3, (0, 0, 255), -1)  # Red for original landmarks

            # Draw transformed landmarks (after scaling and rotation) in green
            for lmk in transformed_landmarks:
                x = int(lmk[0])  # Normalized x-coordinate
                y = int(lmk[1])  # Normalized y-coordinate
                cv2.circle(processed_frame, (x, y), 3, (0, 255, 0), -1)  # Green for transformed landmarks

            # Display the angle indicators on the frame
            processed_frame = draw_angle_indicator(processed_frame, angles)
        
        else:
            cv2.putText(processed_frame, "No Body Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the processed frame with transformations
        cv2.imshow('Combined Video', processed_frame)

        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage with a video file
    video_path = '/Users/macbookair/Downloads/IMG_0505.MOV'
    process_video(video_path)