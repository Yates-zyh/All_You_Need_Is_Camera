import cv2
import numpy as np
import mediapipe as mp
from rotation import get_rotation_matrix, get_scaling_factor
from distance_pose_height import distance_detection
from angle import calculate_3d_angles, draw_angle_indicator


# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# OpenCV for video capture
cap = cv2.VideoCapture(1)  # Use the default camera

# Function to check if the user is still (not moving significantly)
def is_user_still(prev_positions, current_position, threshold=10):
    distance_moved = np.linalg.norm(np.array(prev_positions) - np.array(current_position))
    if distance_moved < threshold:
        return True
    return False

# Function to process the video and give feedback to the user
def process_user_preparation():
    prev_position = None
    is_ready = False
    while not is_ready:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # Provide feedback if the pose is detected
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape

            # Calculate distance between head and feet (for height feedback)
            head = landmarks[mp_pose.PoseLandmark.NOSE]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

            head_to_ankle_distance = np.linalg.norm(np.array([head.x * w, head.y * h]) - 
                                                     np.array([(left_ankle.x + right_ankle.x) / 2 * w, 
                                                              (left_ankle.y + right_ankle.y) / 2 * h]))

            # Give real-time feedback
            distance_feedback = f"Distance from camera: {head_to_ankle_distance:.2f} pixels"

            # Calculate angles
            angles = calculate_3d_angles(results.pose_landmarks.landmark, frame.shape)
            feedback = f"Pitch: {angles['X']:.2f}, Yaw: {angles['Y']:.2f}, Roll: {angles['Z']:.2f}"

            # Show distance and angle feedback on the screen
            cv2.putText(frame, distance_feedback, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, feedback, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Check if the user is still (not moving significantly)
            if prev_position is not None:
                still = is_user_still(prev_position, [head.x * w, head.y * h])
                if still:
                    cv2.putText(frame, "User is still, ready to finish preparation.", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    # If the user is still, allow the "finish preparation"
                    is_ready = True
            prev_position = [head.x * w, head.y * h]

        else:
            cv2.putText(frame, "No Body Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show the processed frame
        cv2.imshow('User Preparation', frame)

        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # After user is ready, capture final position and generate rotation matrix
    print("User preparation finished, calculating rotation matrix...")

    # Generate the rotation matrix using the calculated angles
    rotation_matrix = get_rotation_matrix(angles['X'], angles['Y'], angles['Z'])
    return rotation_matrix

if __name__ == "__main__":
    process_user_preparation()