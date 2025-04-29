import os
import torch
import argparse
import numpy as np
import torchvision
import random
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving figures

# Import functions from single image inference
from singleimage_inference import load_model, preprocess_image, run_inference, CLASSES

def get_random_test_images(test_dir, num_images=20):
    """Get random images from the test directory"""
    image_files = []
    
    # Ensure test_dir is an absolute path
    test_dir = os.path.abspath(test_dir)
    print(f"Looking for images in: {test_dir}")
    
    # Check if directory exists
    if not os.path.exists(test_dir):
        print(f"Directory not found: {test_dir}")
        return image_files
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    
    print(f"Found {len(image_files)} images in total")
    
    # Randomly select images
    if len(image_files) > num_images:
        image_files = random.sample(image_files, num_images)
        
    return image_files

def save_visualization(image_path, result, output_dir, idx):
    """Save visualization of the inference result"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract the class name from the image path
    true_class = os.path.basename(os.path.dirname(image_path))
    
    # Load image
    img = Image.open(image_path)
    
    # Create the visualization
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"True: {true_class}\nPredicted: {result['class_name']}")
    plt.axis('off')
    
    # Display probabilities
    plt.subplot(1, 2, 2)
    probs = result['probabilities']
    y_pos = np.arange(len(CLASSES))
    plt.barh(y_pos, probs)
    plt.yticks(y_pos, CLASSES)
    plt.xlabel('Probability')
    plt.title('Class Probabilities')
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, f"result_{idx:03d}.png")
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def create_video(image_paths, output_video_path, fps=1):
    """Create a video from the visualization images"""
    if not image_paths:
        print("No images to create video")
        return
        
    # Get the first image to determine video size
    first_img = cv2.imread(image_paths[0])
    height, width, _ = first_img.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Add frames to video
    for img_path in image_paths:
        frame = cv2.imread(img_path)
        video.write(frame)
    
    video.release()
    print(f"Video saved to {output_video_path}")

def process_batch(model, image_paths, output_dir, target_size=(224, 224)):
    """Process a batch of images and save visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    result_paths = []
    results_data = []
    
    for idx, image_path in enumerate(image_paths):
        # Run inference
        result = run_inference(model, image_path, target_size)
        
        # Save visualization
        result_path = save_visualization(image_path, result, output_dir, idx)
        result_paths.append(result_path)
        
        # Extract true class from path
        true_class = os.path.basename(os.path.dirname(image_path))
        
        # Store results
        results_data.append({
            'image_path': image_path,
            'true_class': true_class,
            'predicted_class': result['class_name'],
            'correct': true_class == result['class_name'],
            'probabilities': result['class_probabilities']
        })
        
        print(f"Processed image {idx+1}/{len(image_paths)}: {os.path.basename(image_path)}")
        print(f"  True: {true_class}, Predicted: {result['class_name']}")
    
    return result_paths, results_data

def main():
    parser = argparse.ArgumentParser(description='TakuNet Batch Inference')
    parser.add_argument('--test_dir', type=str, default='src/Test', help='Path to test directory')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='predictions_results', help='Output directory for results')
    parser.add_argument('--num_images', type=int, default=20, help='Number of images to process')
    parser.add_argument('--width', type=int, default=224, help='Image width')
    parser.add_argument('--height', type=int, default=224, help='Image height')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Fix path for test directory - remove "src/" prefix if running from src folder
    if os.path.basename(os.getcwd()) == 'src' and args.test_dir.startswith('src/'):
        args.test_dir = args.test_dir[4:]  # Remove 'src/' prefix
    
    # Get random test images
    image_paths = get_random_test_images(args.test_dir, args.num_images)
    
    if not image_paths:
        print(f"No images found in {args.test_dir}")
        return
    
    print(f"Selected {len(image_paths)} images for processing")
    
    # Load model
    model = load_model(args.checkpoint, img_width=args.width, img_height=args.height)
    
    # Process images
    result_paths, results_data = process_batch(
        model, 
        image_paths, 
        args.output_dir, 
        target_size=(args.height, args.width)
    )
    
    # Calculate accuracy
    correct = sum(1 for r in results_data if r['correct'])
    accuracy = correct / len(results_data) if results_data else 0
    print(f"\nResults Summary:")
    print(f"Accuracy: {accuracy:.2f} ({correct}/{len(results_data)})")
    
    # Create video
    video_path = os.path.join(args.output_dir, "results_video.mp4")
    create_video(result_paths, video_path)

if __name__ == "__main__":
    main() 