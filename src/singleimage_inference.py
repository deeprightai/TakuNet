import os
import torch
import argparse
import numpy as np
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

from utils.net_utils import extract_net_params, extract_optim_params, select_arch
from torch import nn

# Define AIDER classes
CLASSES = ["Earthquake", "Fire", "Flood", "Normal"]

def preprocess_image(image_path, target_size):
    """Preprocess an image for TakuNet inference"""
    # Load image using torchvision
    if isinstance(image_path, str):
        image = torchvision.io.read_image(image_path)
    else:
        image = image_path
        
    # Resize
    resize = torchvision.transforms.Resize(target_size)
    image = resize(image)
    
    # Convert to float and normalize
    image = image.float() / 255.0
    
    # Add batch dimension
    image = image.unsqueeze(0)
    return image

def load_model(ckpt_path, num_classes=len(CLASSES), img_width=224, img_height=224):
    """Load the TakuNet model from checkpoint"""
    device = torch.device("cpu")
    
    # Set up model parameters
    net_kwargs = {
        'network': 'takunet',
        'dataset': 'AIDER',
        'input_channels': 3, 
        'output_classes': num_classes,
        'dense': True,
        'stem_reduction': 2,
        'k_folds': 0,
        'split': 'exact',
        'resolution': img_width,
        'classes': CLASSES
    }
    
    optim_kwargs = {
        'optimizer': 'adam',
        'scheduler': 'cosine',
        'batch_size': 1,
        'num_epochs': 100,
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'model_ema': False,
        'dataset_length': 1,
        'scheduler_per_epoch': True,
        'learning_rate_decay': 0.1,
        'learning_rate_decay_steps': 30,
        'min_learning_rate': 1e-6,
        'warmup_epochs': 5,
        'warmup_steps': 1000,
        'weight_decay_end': 0.0001,
        'update_freq': 1,
        'alpha': 0.9,
        'momentum': 0.9
    }
    
    # Create loss function
    criterion = nn.CrossEntropyLoss(reduction="mean")
    
    # Load model from checkpoint
    model = select_arch(net_kwargs, criterion, optim_kwargs, ckpt_path)
    model.eval()
    model.to(device)
    
    return model

def run_inference(model, image_path, target_size=(224, 224)):
    """Run inference on a single image"""
    # Preprocess image
    image = preprocess_image(image_path, target_size)
    
    # Move to device
    image = image.to(next(model.parameters()).device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(image)
        
    # Get probabilities
    probs = torch.nn.functional.softmax(outputs, dim=1)
    
    # Get predicted class
    pred_class = torch.argmax(probs, dim=1).item()
    
    return {
        'class_id': pred_class,
        'class_name': CLASSES[pred_class],
        'probabilities': probs[0].cpu().numpy(),
        'class_probabilities': {CLASSES[i]: probs[0][i].item() for i in range(len(CLASSES))}
    }

def visualize_result(image_path, result):
    """Visualize the inference result"""
    # Load and display image
    img = Image.open(image_path)
    plt.figure(figsize=(12, 8))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Prediction: {result['class_name']}")
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
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='TakuNet Inference')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--width', type=int, default=224, help='Image width')
    parser.add_argument('--height', type=int, default=224, help='Image height')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.checkpoint, img_width=args.width, img_height=args.height)
    
    # Run inference
    result = run_inference(model, args.image, target_size=(args.height, args.width))
    
    # Print results
    print(f"Predicted class: {result['class_name']}")
    print("Class probabilities:")
    for class_name, prob in result['class_probabilities'].items():
        print(f"  {class_name}: {prob:.4f}")
    
    # Visualize if requested
    if args.visualize:
        visualize_result(args.image, result)

if __name__ == "__main__":
    main()