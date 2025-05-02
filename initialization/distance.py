import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize OpenCV for camera input
cap = cv2.VideoCapture(1)  # Change to 0 if using the default camera

# Function to calculate the distance between the user and the camera (based on face or body keypoints)
def get_distance_from_camera(keypoint, frame):
    center = (frame.shape[1] // 2, frame.shape[0] // 2)
    distance = np.linalg.norm(np.array([keypoint[0] - center[0], keypoint[1] - center[1]]))
    return distance

# Real-world Distance Calculation (from pixels to meters)
def convert_pixels_to_meters(pixel_distance, focal_length, sensor_size, image_resolution):
    sensor_size_in_pixels = image_resolution[0]
    real_distance = (pixel_distance * focal_length * sensor_size) / sensor_size_in_pixels
    return real_distance  # In meters

# Function to check if the user is within the optimal camera range (1.5 - 3 meters)
def check_distance_from_camera(keypoint, frame):
    distance = get_distance_from_camera(keypoint, frame)
    
    # Set a threshold for optimal distance (this is a rough approximation)
    if distance < 100:  # Too close
        return f"Move Back | Distance: {distance:.2f} pixels"
    elif distance > 200:  # Too far
        return f"Move Closer | Distance: {distance:.2f} pixels"
    else:
        return f"Good Position | Distance: {distance:.2f} pixels"

# Function to process the video stream
def process_video_stream():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB (MediaPipe requires RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform pose detection
        results = pose.process(frame_rgb)
        
        feedback = "Position yourself within the optimal range (1.5 - 3 meters)."
        if results.pose_landmarks:
            # Draw landmarks on the frame
            for landmark in results.pose_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Draw the landmarks (red dots)
            
            # Get a keypoint (e.g., wrist) for checking distance
            right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            feedback = check_distance_from_camera([right_wrist.x * frame.shape[1], right_wrist.y * frame.shape[0]], frame)
        
        # Show feedback on the screen
        cv2.putText(frame, feedback, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show the frame
        cv2.imshow('Dance Map Calibration', frame)
        
        # Break the loop if 'q' is pressed
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