#import sys
#sys.path.append('/Users/macbookair/Depth-Anything')
#sys.path.append('/Users/macbookair/Depth-Anything-V2')
import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import numpy as np
#from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

# Initialize model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
depth_anything = DepthAnything.from_pretrained("LiheYoung/depth_anything_vitl14").to(device).eval()

# Preprocessing transformations
transform = transforms.Compose([
    Resize(518, 518, resize_target=False, keep_aspect_ratio=True, ensure_multiple_of=14, image_interpolation_method=cv2.INTER_CUBIC),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet()
])

def get_body_roi(frame, keypoints):
    """
    Get the Region of Interest (ROI) of the human body using pose keypoints.
    keypoints should be a list of (x, y) tuples for body parts like shoulders, hips, etc.
    """
    # Get the bounding box of the body from the keypoints
    x_min = min([keypoints[i][0] for i in range(len(keypoints))])
    x_max = max([keypoints[i][0] for i in range(len(keypoints))])
    y_min = min([keypoints[i][1] for i in range(len(keypoints))])
    y_max = max([keypoints[i][1] for i in range(len(keypoints))])

    # Crop the image to the ROI
    roi = frame[y_min:y_max, x_min:x_max]
    return roi

def calculate_distance_from_roi(roi, depth_map):
    """
    Calculate the distance of the human body from the camera using the depth map.
    The ROI should be the cropped region of the human body.
    """
    # Ensure the depth_map is 2D (single channel depth map)
    if len(depth_map.shape) == 3:  # If it's a color image, get the first channel (grayscale depth map)
        depth_map = depth_map[:, :, 0]
    
    # Ensure the ROI mask is binary and of the same size as the depth map
    mask = np.zeros_like(depth_map)
    mask[roi > 0] = 1  # Create a binary mask from the ROI (ensure ROI is the correct shape)
    
    # Apply the mask to the depth map to extract the depth values for the body region
    roi_depth = depth_map * mask
    
    # Calculate the mean depth within the ROI
    roi_depth_values = roi_depth[roi_depth > 0]  # Get non-zero depth values within the ROI
    if len(roi_depth_values) > 0:
        distance = np.mean(roi_depth_values)  # The mean depth of the ROI as the person's distance
    else:
        distance = -1  # In case there's no valid depth in the ROI
    
    print(f"Average Depth in ROI: {distance}")
    return distance

def process_frame(frame, transform, model, device):
    # Preprocess the frame
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
    h, w = image.shape[:2]
    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).to(device)

    # Predict depth using the model
    with torch.no_grad():
        depth = model(image)
    #print(f"Depth: {depth}")
    # the center of the image is the most important part
    #print(f"Depth shape: {depth.shape}")
    ww = depth.shape[2]
    hh = depth.shape[1]
    #dis = depth[0][hh // 2][ww // 2]
    #print(f"Depth at center: {dis}")
    # Post-process the depth map
    depth = torch.nn.functional.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    #print(f"Depth shape: {depth.shape}")
    dis = depth[h // 2][w // 2]
    print(f"Depth at center: {dis}")
    print(f"Depth at foot: {depth[h // 2][w // 4]}")
    print(f"Depth at head: {depth[h // 2][3*w // 4]}")
    print(f"bottom right corner: {depth[0][0]}")
    print(f"top left corner: {depth[h-1][w-1]}")
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.cpu().numpy().astype(np.uint8)

    return depth

def process_camera(camera_index, transform, model, device):
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Here, detect keypoints using a pose detection model (e.g., YOLOv8-Pose)
        # For simplicity, we'll assume `keypoints` is already provided
        keypoints = [(100, 150), (200, 250), (150, 300)]  # Example keypoints for demonstration
        
        # Process the frame to get depth
        depth_map = process_frame(frame, transform, model, device)

        # Get the body region of interest
        body_roi = get_body_roi(frame, keypoints)

        # Calculate the distance based on the ROI
        distance = calculate_distance_from_roi(body_roi, depth_map)

        # Display the distance on the frame
        distance_text = f"Distance: {distance:.2f} meters"
        cv2.putText(frame, distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show the frame with the distance
        cv2.imshow("Frame with Distance", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def process_video(video_path, transform, model, device):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        depth = process_frame(frame, transform, model, device)
        #print(f"Depth: {depth}")
        
        # Visualize or process the depth map
        depth_colored = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        
        cv2.imshow("Depth Map", depth_colored)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def process_images(image_path, transform, model, device):
    filenames = os.listdir(image_path)
    filenames = [os.path.join(image_path, filename) for filename in filenames if not filename.startswith('.')]
    
    for filename in tqdm(filenames):
        raw_image = cv2.imread(filename)
        depth = process_frame(raw_image, transform, model, device)
        
        depth_colored = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        cv2.imwrite(f"./output/{os.path.basename(filename)}", depth_colored)

def main():
    # Set the parameters here
    img_path = "/Users/macbookair/Downloads/IMG_0506.mov"  # Set the path for images or videos
    #img_path = "/Users/macbookair/Downloads/img"  # Set the path for images or videos
    camera_index = None  # Set to the camera index, 0 is the default camera
    outdir = './output'  # Output directory
    encoder = 'vitl'  # Encoder option (could be 'vits', 'vitb', 'vitl')

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load the model
    depth_anything = DepthAnything.from_pretrained(f'LiheYoung/depth_anything_{encoder}14').to(DEVICE).eval()

    transform = transforms.Compose([
        Resize(518, 518, resize_target=False, keep_aspect_ratio=True, ensure_multiple_of=14, image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet()
    ])

    # Check for camera input
    if camera_index is not None:
        process_camera(camera_index, transform, depth_anything, DEVICE)
    elif os.path.isfile(img_path):
        if img_path.endswith(('.mp4', '.avi', '.mov')):
            process_video(img_path, transform, depth_anything, DEVICE)
        else:
            print(f"Error: {img_path} is not a valid video file.")
    elif os.path.isdir(img_path):
        process_images(img_path, transform, depth_anything, DEVICE)
    else:
        print("Error: Invalid input path.")

if __name__ == '__main__':
    main()