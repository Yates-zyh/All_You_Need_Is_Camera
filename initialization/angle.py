import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# OpenCV for video capture
cap = cv2.VideoCapture(1)  # Change to 0 if using the default camera

# Function to calculate angle between two points
def calculate_angle(p1, p2):
    delta_x = p2[0] - p1[0]
    delta_y = p2[1] - p1[1]
    angle = np.degrees(np.arctan2(delta_y, delta_x))
    return angle

# Function to process the video stream
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

            # Draw results on the frame
            cv2.putText(frame, f"Pitch: {pitch_angle:.2f} degrees", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"Yaw: {yaw_angle:.2f} degrees", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Visualize key points
            for landmark in landmarks:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        # Show the frame
        cv2.imshow('Camera Angle Detection', frame)

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