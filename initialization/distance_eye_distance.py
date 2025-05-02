'''
not working for distance estimation beyond 1m
'''
import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Camera parameters (adjust these based on your camera or video)
FOCAL_LENGTH = 600  # In pixels (approximate for a typical webcam)
IMAGE_WIDTH = 640   # In pixels (adjust based on your video resolution)

# Real-world reference: average distance between eyes (in meters)
REAL_EYE_DISTANCE = 0.063  # 6.3 cm (average for adults)

# Function to calculate pixel distance between two keypoints
def get_pixel_distance(keypoint1, keypoint2):
    return np.linalg.norm(np.array(keypoint1) - np.array(keypoint2))

# Function to estimate distance from camera using inter-eye distance
def distance_detection(frame_rgb):
    """
    Estimate distance from camera using facial landmarks in a single RGB frame.
    
    Args:
        frame_rgb (numpy.ndarray): RGB frame from video or webcam.
    
    Returns:
        tuple: (processed BGR frame, distance in meters or None, status message)
    """
    # Convert RGB frame to BGR for OpenCV display
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    h, w, _ = frame.shape

    # Process frame with MediaPipe Face Mesh
    results = face_mesh.process(frame_rgb)

    distance = None
    distance_text = "No Face Detected"
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get landmarks for left and right eyes (using MediaPipe Face Mesh indices)
            left_eye = face_landmarks.landmark[33]  # Left eye outer corner
            right_eye = face_landmarks.landmark[263]  # Right eye outer corner

            # Convert normalized coordinates to pixel coordinates
            left_eye_px = (int(left_eye.x * w), int(left_eye.y * h))
            right_eye_px = (int(right_eye.x * w), int(right_eye.y * h))

            # Calculate pixel distance between eyes
            pixel_eye_distance = get_pixel_distance(left_eye_px, right_eye_px)

            # Estimate real-world distance
            if pixel_eye_distance == 0:  # Avoid division by zero
                distance_text = "Error: Zero pixel distance detected"
            else:
                distance = (REAL_EYE_DISTANCE * FOCAL_LENGTH) / pixel_eye_distance

                # Check if the user is within the optimal camera range (1.5 - 3 meters)
                if distance < 1.5:  # Too close
                    distance_text = f"Move Back | Distance: {distance:.2f} m"
                elif distance > 3.0:  # Too far
                    distance_text = f"Move Closer | Distance: {distance:.2f} m"
                else:
                    distance_text = f"Good Position | Distance: {distance:.2f} m"

            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

    # Overlay distance text on the frame
    cv2.putText(frame, distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame, distance, distance_text

# Main function for video processing demo
def main():
    # Open the recorded video file
    video_path = '/Users/macbookair/Downloads/IMG_0508.MOV'  # Specify your video file here
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Set up video writer for output
    output_path = 'output_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame")
            break

        # Convert frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame to estimate distance
        processed_frame, distance, distance_text = distance_detection(frame_rgb)

        # Write the processed frame to the output video
        out.write(processed_frame)

        # Display the processed frame
        cv2.imshow("Distance Estimation", processed_frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    face_mesh.close()

if __name__ == "__main__":
    main()