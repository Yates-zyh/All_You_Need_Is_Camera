import cv2
import mediapipe as mp
import numpy as np
from distance import get_distance_from_camera, convert_pixels_to_meters  # Importing distance functions
from angle import calculate_angle  # Importing angle calculation function

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# OpenCV for video capture
cap = cv2.VideoCapture(1)  # Change to 0 if using the default camera

# Function to map the current position to the ideal "perfect" position
def map_to_standard_position(current_pitch, current_yaw, current_distance):
    # Ideal standard position:
    ideal_pitch = 0.0  # Perfect pitch (no up/down tilt)
    ideal_yaw = 0.0  # Perfect yaw (directly facing the camera)
    ideal_distance = 2.0  # Example standard distance (in meters)

    # Allow for small adjustments (threshold)
    pitch_threshold = 5.0  # Allow a 5-degree deviation for pitch
    yaw_threshold = 5.0    # Allow a 5-degree deviation for yaw

    # Calculate the necessary adjustments for pitch, yaw, and distance
    pitch_adjustment = np.clip(ideal_pitch - current_pitch, -pitch_threshold, pitch_threshold)
    yaw_adjustment = np.clip(ideal_yaw - current_yaw, -yaw_threshold, yaw_threshold)
    distance_adjustment = ideal_distance - current_distance

    return pitch_adjustment, yaw_adjustment, distance_adjustment

# Function to process the video stream and calculate angles and adjustments
def process_video_stream():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            # Get key points
            landmarks = results.pose_landmarks.landmark
            nose = [landmarks[mp_pose.PoseLandmark.NOSE].x * frame.shape[1], landmarks[mp_pose.PoseLandmark.NOSE].y * frame.shape[0]]
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame.shape[1], landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame.shape[0]]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame.shape[1], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame.shape[0]]

            # Calculate pitch (up/down angle) from nose to horizontal line
            pitch_angle = calculate_angle([frame.shape[1] // 2, frame.shape[0]], nose)

            # Calculate yaw (left/right angle) from shoulder symmetry
            yaw_angle = calculate_angle(left_shoulder, right_shoulder)

            # Map to perfect position (0° pitch, 0° yaw)
            pitch_adjustment, yaw_adjustment = map_to_perfect_position(pitch_angle, yaw_angle)

            # Provide feedback for how much adjustment is needed
            feedback = f"Pitch Adjustment: {pitch_adjustment:.2f} degrees | Yaw Adjustment: {yaw_adjustment:.2f} degrees"

            # Draw results on the frame
            cv2.putText(frame, feedback, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Visualize key points
            for landmark in landmarks:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        # Show the frame
        cv2.imshow('Camera Angle Adjustment', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Main function to run the process
def main():
    process_video_stream()

if __name__ == "__main__":
    main()