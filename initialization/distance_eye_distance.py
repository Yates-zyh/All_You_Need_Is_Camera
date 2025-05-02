'''
does not work for far distances
'''
import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Camera parameters (adjust these based on your camera)
FOCAL_LENGTH = 600  # In pixels (approximate for a typical webcam)
SENSOR_WIDTH = 0.0064  # In meters (6.4 mm, typical for webcams)
IMAGE_WIDTH = 640  # In pixels (adjust based on your camera resolution)

# Real-world reference: average distance between eyes (in meters)
REAL_EYE_DISTANCE = 0.063  # 6.3 cm (average for adults)

# Function to calculate pixel distance between two keypoints
def get_pixel_distance(keypoint1, keypoint2):
    return np.linalg.norm(np.array(keypoint1) - np.array(keypoint2))

# Function to estimate real-world distance from camera using inter-eye distance
def estimate_distance_from_camera(pixel_eye_distance):
    # Using the pinhole camera model: distance = (real_size * focal_length) / pixel_size
    if pixel_eye_distance == 0:  # Avoid division by zero
        return float('inf')
    distance = (REAL_EYE_DISTANCE * FOCAL_LENGTH) / pixel_eye_distance
    return distance  # In meters

# Function to check if the user is within the optimal camera range (1.5 - 3 meters)
def check_distance_from_camera(distance):
    if distance < 1.5:  # Too close
        return f"Move Back | Distance: {distance:.2f} m"
    elif distance > 3.0:  # Too far
        return f"Move Closer | Distance: {distance:.2f} m"
    else:
        return f"Good Position | Distance: {distance:.2f} m"

# Main function for real-time distance estimation
def main():
    # Open webcam
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        distance_text = "No Face Detected"
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get landmarks for left and right eyes (using MediaPipe Face Mesh indices)
                left_eye = face_landmarks.landmark[33]  # Left eye outer corner
                right_eye = face_landmarks.landmark[263]  # Right eye outer corner

                # Convert normalized coordinates to pixel coordinates
                h, w, _ = frame.shape
                left_eye_px = (int(left_eye.x * w), int(left_eye.y * h))
                right_eye_px = (int(right_eye.x * w), int(right_eye.y * h))

                # Calculate pixel distance between eyes
                pixel_eye_distance = get_pixel_distance(left_eye_px, right_eye_px)

                # Estimate real-world distance
                distance = estimate_distance_from_camera(pixel_eye_distance)

                # Check if the user is in the optimal range
                distance_text = check_distance_from_camera(distance)

                # Draw landmarks and distance text on the frame
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
                cv2.putText(frame, distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Distance Estimation", frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

if __name__ == "__main__":
    main()