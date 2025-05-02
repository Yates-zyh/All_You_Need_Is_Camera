'''
not working for distance estimation beyond 1m
'''
import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Constants for distance calculation
REAL_FACE_WIDTH = 0.15  # Average face width in meters (15cm)
FOCAL_LENGTH = 1000     # Pre-calibrated focal length in pixels

def distance_detection(frame_rgb):
    """Detect faces and calculate distance from camera."""
    results = face_detection.process(frame_rgb)
    annotated_frame = frame_rgb.copy()
    distance = None

    if results.detections:
        # Get first face detection
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        
        # Calculate face width in pixels
        face_width_px = bbox.width * frame_rgb.shape[1]
        
        # Calculate distance using face width
        if face_width_px > 0:
            distance = (REAL_FACE_WIDTH * FOCAL_LENGTH) / face_width_px
        
        # Draw bounding box and info
        mp.solutions.drawing_utils.draw_detection(annotated_frame, detection)
        
        # Convert back to BGR for OpenCV display
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        
        # Add distance text
        if distance is not None:
            cv2.putText(annotated_frame, f"Distance: {distance:.2f}m", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
    
    return annotated_frame, distance

# Main video processing function
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame for distance detection
        processed_frame, distance = distance_detection(frame_rgb)
        
        # Display results
        cv2.imshow('Face Distance Detection', processed_frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Run the demo with your video file
video_path = '/Users/macbookair/Downloads/IMG_0508.MOV'
process_video(video_path)