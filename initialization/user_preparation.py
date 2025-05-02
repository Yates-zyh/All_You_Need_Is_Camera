import glob
import cv2
import mediapipe as mp
import numpy as np
from distance import get_distance_from_camera, convert_pixels_to_meters  # Importing distance functions
from angle import calculate_angle  # Importing angle calculation function
from position_mapping import map_to_standard_position  # Importing position mapping function

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# OpenCV for video capture
cap = cv2.VideoCapture(0)  # Use the default camera

# Function to check if the user is still (not moving significantly)
def is_user_still(prev_positions, current_position, threshold=10):
    distance_moved = np.linalg.norm(np.array(prev_positions) - np.array(current_position))
    if distance_moved < threshold:
        return True
    return False

# Function to draw the adjustment vectors (arrows) to visualize the movement
def draw_adjustment_arrows(frame, original_keypoints, adjusted_keypoints):
    for orig, adjusted in zip(original_keypoints, adjusted_keypoints):
        # Convert the points to integers before drawing the arrows
        orig = tuple(map(int, orig))
        adjusted = tuple(map(int, adjusted))
        
        cv2.arrowedLine(frame, orig, adjusted, (0, 255, 0), 2)

# Main function to capture the posture when the user is still
def main():
    prev_positions = [None, None]  # To store the previous positions of key points (e.g., nose, shoulders)
    user_still_time = 0  # Time user has been still
    capture_threshold = 3  # Threshold in seconds for capturing the posture info

    # Camera parameters for distance calculation (example values, adjust as needed)
    focal_length = 1000  # Example focal length (in pixels)
    sensor_size = 4.0  # Example sensor size (in mm)
    image_resolution = (640, 480)  # Example resolution (width x height)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get key points (e.g., nose, shoulders)
            nose = [landmarks[mp_pose.PoseLandmark.NOSE].x * frame.shape[1], landmarks[mp_pose.PoseLandmark.NOSE].y * frame.shape[0]]
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame.shape[1], landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame.shape[0]]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame.shape[1], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame.shape[0]]

            # Calculate pitch (up/down angle) from nose to horizontal line
            pitch_angle = calculate_angle([frame.shape[1] // 2, frame.shape[0]], nose)

            # Calculate yaw (left/right angle) from shoulder symmetry
            yaw_angle = calculate_angle(left_shoulder, right_shoulder)

            # Calculate the distance from the camera
            pixel_distance = get_distance_from_camera(nose, frame)
            real_distance = convert_pixels_to_meters(pixel_distance, focal_length, sensor_size, image_resolution)

            # Check if the user is still (not moving significantly)
            current_position = [nose[0], nose[1]]  # Store the current position (e.g., nose)
            if prev_positions[0] is not None and is_user_still(prev_positions, current_position):
                user_still_time += 1  # Increment the time user has been still

                # If the user has been still for enough time, capture their posture info
                if user_still_time >= capture_threshold:
                    feedback = f"User is still. Pitch: {pitch_angle:.2f} degrees | Yaw: {yaw_angle:.2f} degrees | Distance: {real_distance:.2f} meters"
                    print(feedback)

                    # Map to the standard position
                    pitch_adjustment, yaw_adjustment, distance_adjustment = map_to_perfect_position(pitch_angle, yaw_angle, real_distance)

                    feedback_adjustment = f"Adjust Pitch: {pitch_adjustment:.2f} degrees | Adjust Yaw: {yaw_adjustment:.2f} degrees | Adjust Distance: {distance_adjustment:.2f} meters"
                    print(feedback_adjustment)

                    user_still_time = 0  # Reset the time after capturing the info

            # Update previous positions for the next iteration
            prev_positions = [nose[0], nose[1]]

            # Visualization: Show original posture and adjusted (converted) posture
            # 1. Original Posture (on the first window)
            cv2.putText(frame, f"Original Pitch: {pitch_angle:.2f} degrees", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"Original Yaw: {yaw_angle:.2f} degrees", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"Original Distance: {real_distance:.2f} meters", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Create a copy of the frame for converted position visualization
            converted_frame = frame.copy()

            # 2. Converted Position (on the second window)
            pitch_adjustment, yaw_adjustment, distance_adjustment = map_to_perfect_position(pitch_angle, yaw_angle, real_distance)
            cv2.putText(converted_frame, f"Converted Pitch: {pitch_angle + pitch_adjustment:.2f} degrees", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(converted_frame, f"Converted Yaw: {yaw_angle + yaw_adjustment:.2f} degrees", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(converted_frame, f"Converted Distance: {real_distance + distance_adjustment:.2f} meters", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Visualize the keypoints for both the original and converted positions
            original_keypoints = [(landmark.x * frame.shape[1], landmark.y * frame.shape[0]) for landmark in landmarks]
            adjusted_keypoints = [(x + pitch_adjustment, y + yaw_adjustment) for x, y in original_keypoints]

            # Draw adjustment arrows
            draw_adjustment_arrows(converted_frame, original_keypoints, adjusted_keypoints)

            # Show both frames
            cv2.imshow('Original Posture', frame)
            cv2.imshow('Converted Posture', converted_frame)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()