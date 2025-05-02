'''
The initialization module is responsible for setting up the camera and its parameters.
It includes functions to initialize the camera, set its resolution, and adjust its settings.
The module also includes functions to check if the camera is ready and to provide feedback to the user
during the initialization process.
should return the camera object and any other necessary parameters for further processing.
'''
import cv2
import numpy as np
import mediapipe as mp

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Pose
        results = pose.process(frame_rgb)

        # Draw landmarks on the frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the frame
        cv2.imshow('Pose Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    # Initialize MediaPipe pose detection
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    # Example usage
    video_path = '/Users/macbookair/Downloads/IMG_0505.MOV'
    process_video(video_path)
