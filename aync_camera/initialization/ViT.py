import torch
import cv2
import numpy as np
from transformers import ViTFeatureExtractor, ViTForImageClassification

class ViTInitializer:
    def __init__(self, model_name: str, device: str = 'cpu'):
        """
        Initialize the ViT model with the given parameters.

        Args:
        - model_name (str): The name of the pre-trained ViT model (e.g., 'google/vit-base-patch16-224-in21k').
        - device (str): The device to run the model on ('cpu' or 'cuda').
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.feature_extractor = None

    def initialize(self):
        """
        Initialize the ViT model and the feature extractor.
        """
        # Load the pre-trained model for depth estimation or similar task
        self.model = ViTForImageClassification.from_pretrained(self.model_name).to(self.device)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(self.model_name)

    def predict_distance(self, frame):
        """
        Predict the distance from the camera to the person using ViT.

        Args:
        - frame (numpy array): The input image/frame.

        Returns:
        - float: The predicted distance.
        """
        # Preprocess the frame
        inputs = self.feature_extractor(images=frame, return_tensors="pt").to(self.device)
        
        # Make a prediction using the ViT model
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Assuming the model returns logits with a shape of (batch_size, num_classes)
        print("Model Output Logits:", outputs.logits)
        
        # Get the index of the maximum logit value (or use it directly as the predicted class)
        distance = torch.max(outputs.logits, dim=1).values.item()  # Extract the value of the max logit
        return distance

    def predict(self, frame):
        """
        Predict the distance from the frame.

        Args:
        - frame (numpy array): The input image/frame.

        Returns:
        - float: The predicted distance.
        """
        distance = self.predict_distance(frame)
        return {'distance': distance}


def main():
    # Initialize the ViT model for distance estimation
    # no trained model available for depth estimation yet
    model_name = "google/vit-base-patch16-224-in21k"  # need to be Replaced with a suitable depth estimation model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vit_initializer = ViTInitializer(model_name, device)
    vit_initializer.initialize()

    # Open the webcam
    cap = cv2.VideoCapture(1)  # 0 for default camera

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break
        
        # Resize the frame to match the input size required by the ViT model
        frame_resized = cv2.resize(frame, (224, 224))  # Resize frame to (224, 224) for ViT
        
        # Predict the distance
        result = vit_initializer.predict(frame_resized)

        # Display the predicted distance on the frame
        distance = result['distance']
        cv2.putText(frame, f"Predicted Distance: {distance:.2f}m", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the frame with the predicted distance
        cv2.imshow("Camera Feed", frame)

        # Break the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()