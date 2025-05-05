import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to calculate pixel distance between two keypoints
def get_pixel_distance(keypoint1, keypoint2):
    return np.linalg.norm(np.array(keypoint1) - np.array(keypoint2))

def get_pixel_head_toe(landmarks):
    """Calculate the distance between head and toe landmarks."""
    head = np.array([landmarks[mp_pose.PoseLandmark.NOSE].x, landmarks[mp_pose.PoseLandmark.NOSE].y])
    toe = np.array([landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y])
    distance = np.linalg.norm(head - toe)
    return distance

def distance_detection(frame_rgb, height=1.7, focal_length=600):
    """
    Estimate distance from camera using pose landmarks in a single RGB frame.
    
    Args:
        frame_rgb (numpy.ndarray): RGB frame from video or webcam.
    
    Returns:
        tuple: (processed BGR frame, distance in meters or None, status message)
    """
    # Convert RGB frame to BGR for OpenCV display
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    h, w, _ = frame.shape

    # Process frame with MediaPipe Pose
    results = pose.process(frame_rgb)

    distance = None
    distance_text = "No body Detected"

    if results.pose_landmarks:
        # Access the landmarks (this is a list of keypoints)
        landmarks = results.pose_landmarks.landmark

        # Get landmarks for head and toe (using MediaPipe Pose indices)
        head = landmarks[mp_pose.PoseLandmark.NOSE]
        toe = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]

        # Convert normalized coordinates to pixel coordinates
        head_px = (int(head.x * w), int(head.y * h))
        toe_px = (int(toe.x * w), int(toe.y * h))

        # Calculate pixel distance between head and toe
        pixel_distance = get_pixel_distance(head_px, toe_px)

        # Estimate real-world distance
        if pixel_distance == 0:
            distance_text = "Error: Zero pixel distance detected"
        else:
            distance = (height * focal_length) / pixel_distance

            # Check if the user is within the optimal camera range (1.5 - 3 meters)
            if distance < 1.5:  # Too close
                distance_text = f"Move Back | Distance: {distance:.2f} m"
            elif distance > 3.0:  # Too far
                distance_text = f"Move Closer | Distance: {distance:.2f} m"
            else:
                distance_text = f"Good Position | Distance: {distance:.2f} m"

        # Draw landmarks on the frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # Overlay distance text on the frame
    cv2.putText(frame, distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame, distance, distance_text

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

# Main video processing function
def main(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame for distance detection
        processed_frame, distance, distance_text = distance_detection(frame_rgb, height=1.7, focal_length=600)
        
        # Display results
        cv2.imshow('Distance Detection', processed_frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage with a video file
    video_path = '/Users/macbookair/Downloads/IMG_0505.MOV'
    main(video_path)