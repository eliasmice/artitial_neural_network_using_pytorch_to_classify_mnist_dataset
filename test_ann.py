# -*- coding: utf-8 -*-
"""
Test script for CNN model evaluation on MNIST and student handwritten data.

Usage:
    python test_ann.py --dataset mnist     # Test on MNIST dataset (default)
    python test_ann.py --dataset students  # Test on student handwritten data
    
Requirements:
    - For student data: Data.npy and Labels.npy files should be in current directory
    - For MNIST: Will be downloaded automatically if not present
    - Model file: mnist_cnn_best.pt should be in current directory
    - Config file: best_model_config_focused.json should be in current directory
"""

print("Importing packages...")
import torch
import torch.nn.functional as F
import wandb
from sklearn.metrics import confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import json
import os
import argparse
import time

from ann import CNN

# Function to load student handwritten data
def load_student_data(data_path="."):
    """
    Load handwritten digit data from students.
    Expects Data.npy and Labels.npy files in the current directory.
    """
    try:
        # Load numpy arrays from current directory
        data_file = os.path.join(data_path, "Data.npy")
        labels_file = os.path.join(data_path, "Labels.npy")
        
        if not os.path.exists(data_file) or not os.path.exists(labels_file):
            print(f"Warning: Student data files not found!")
            print(f"Expected: Data.npy and Labels.npy in {data_path}")
            return None, None
        
        # Load data and labels
        data = np.load(data_file)
        labels = np.load(labels_file)
        
        print(f"Loaded student data shape: {data.shape}")
        print(f"Loaded student labels shape: {labels.shape}")
        print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")
        print(f"Unique labels: {np.unique(labels)}")
        
        # Convert to torch tensors
        # Assuming data is in range [0, 255] and needs normalization
        if data.max() > 1.0:
            data = data.astype(np.float32) / 255.0
        
        # Ensure data is in correct shape for CNN: [N, 1, 28, 28]
        if len(data.shape) == 3:  # [N, 28, 28]
            data = data[:, np.newaxis, :, :]  # Add channel dimension
        elif len(data.shape) == 2:  # [N, 784] - flattened
            data = data.reshape(-1, 1, 28, 28)
        
        # Normalize using MNIST statistics
        mean = 0.1307
        std = 0.3081
        data = (data - mean) / std
        
        # Convert to torch tensors
        images = torch.from_numpy(data).float()
        labels = torch.from_numpy(labels).long()
        
        print(f"Final tensor shapes: images {images.shape}, labels {labels.shape}")
        print(f"Loaded {len(images)} student handwritten digits")
        
        return images, labels
        
    except Exception as e:
        print(f"Error loading student data: {e}")
        return None, None

# Function to load configuration from JSON and fix potential inconsistencies
def load_and_fix_config(config_file, model_file):
    """
    Load configuration from JSON and detect if saved model has different BatchNorm setting.
    """
    with open(config_file, 'r') as f:
        data = json.load(f)
    
    config = data['config']
    
    # Convert types as needed
    if isinstance(config.get('conv_channels'), list):
        config['conv_channels'] = tuple(config['conv_channels'])
    
    if isinstance(config.get('fc_units'), list):
        config['fc_units'] = tuple(config['fc_units'])
    
    # Convert string "False"/"True" to boolean if needed
    if isinstance(config.get('use_batch_norm'), str):
        config['use_batch_norm'] = config['use_batch_norm'].lower() == 'true'
    
    # Check if the saved model actually has BatchNorm regardless of config
    try:
        state_dict = torch.load(model_file, map_location='cpu')
        has_batch_norm = any('bn' in key for key in state_dict.keys())
        
        if has_batch_norm and not config['use_batch_norm']:
            print("WARNING: JSON says use_batch_norm=False but model has BatchNorm layers.")
            print("Adjusting configuration to match saved model...")
            config['use_batch_norm'] = True
        elif not has_batch_norm and config['use_batch_norm']:
            print("WARNING: JSON says use_batch_norm=True but model has no BatchNorm layers.")
            print("Adjusting configuration to match saved model...")
            config['use_batch_norm'] = False
            
    except Exception as e:
        print(f"Warning: Could not verify BatchNorm in model file: {e}")
    
    return config

# Parse command line arguments
parser = argparse.ArgumentParser(description='Test CNN on MNIST and/or student data')
parser.add_argument('--dataset', choices=['mnist', 'students'], default='mnist',
                    help='Which dataset to evaluate: mnist or students')
args = parser.parse_args()

# Initialize wandb for evaluation
wandb.init(
    project="mnist-cnn-evaluation",
    name=f"cnn-test-evaluation-{args.dataset}",
    config={
        "model_type": "CNN",
        "dataset": args.dataset,
        "evaluation": "final_test"
    }
)

# Load datasets based on arguments
print(f"Selected dataset: {args.dataset}")

if args.dataset == 'mnist':
    # Load MNIST test dataset
    print("Loading MNIST test dataset...")
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    dataset_name = 'MNIST Test'
    data_loader = test_loader

elif args.dataset == 'students':
    # Load student handwritten data
    print("Loading student handwritten data...")
    student_images, student_labels = load_student_data(".")
    if student_images is None:
        print("Failed to load student data. Exiting...")
        exit(1)
    
    student_dataset = torch.utils.data.TensorDataset(student_images, student_labels)
    student_loader = torch.utils.data.DataLoader(student_dataset, batch_size=100, shuffle=False)
    dataset_name = 'Student Handwritten'
    data_loader = student_loader

# Load and fix model configuration
print("Loading model configuration...")
model_config = load_and_fix_config("best_model_config_focused.json", "mnist_cnn_best.pt")
print("Final configuration:", model_config)

# Create model with correct configuration
print("Creating CNN model...")
model = CNN(config=model_config)
print(f"Model architecture:")
print(f"  Conv channels: {model.conv_channels}")
print(f"  FC units: {model.fc_units}")
print(f"  Dropout rate: {model.dropout_rate}")
print(f"  Use BatchNorm: {model.use_batch_norm}")
print(f"  Activation: {model.activation_name}")

# Load trained model weights
print("Loading trained model weights...")
model.load("mnist_cnn_best.pt")

# Configure device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Evaluation function
def evaluate_model(data_loader, dataset_name):
    """
    Evaluate the model on a specific dataset and compute metrics.
    """
    print(f"\nStarting evaluation on {dataset_name}...")
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            # Move data to device
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            
            # Calculate loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            
            # Get predictions
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            # Store predictions and targets for detailed analysis
            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy())
    
    # Calculate overall metrics
    test_loss /= len(data_loader.dataset)
    accuracy = 100. * correct / len(data_loader.dataset)
    
    # Calculate precision and recall
    precision = precision_score(all_targets, all_preds, average='macro')
    recall = recall_score(all_targets, all_preds, average='macro')
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    # Calculate per-class metrics
    precision_per_class = precision_score(all_targets, all_preds, average=None)
    recall_per_class = recall_score(all_targets, all_preds, average=None)
    
    # Generate confusion matrix
    conf_mat = confusion_matrix(all_targets, all_preds)
    
    # Close any existing figures to ensure only one is created
    plt.close('all')
    
    # Create a single figure for the confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=range(10))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title(f"CNN Confusion Matrix - {dataset_name}")
    plt.tight_layout()
    
    # Generate unique filename for each run to avoid caching issues
    timestamp = int(time.time())
    dataset_clean = dataset_name.replace(' ', '_').lower()
    confusion_matrix_file = f'cnn_confusion_matrix_{dataset_clean}_{timestamp}.png'
    
    # Save with explicit flushing to ensure file is written completely
    plt.savefig(confusion_matrix_file, dpi=150, bbox_inches='tight')
    plt.draw()  # Force drawing
    fig.canvas.flush_events()  # Flush any pending GUI events
    
    # Small delay to ensure file is completely written
    time.sleep(0.5)
    
    # Print results
    print("\n" + "="*60)
    print(f"{dataset_name.upper()} EVALUATION RESULTS")
    print("="*60)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Test Precision (macro): {precision*100:.2f}%")
    print(f"Test Recall (macro): {recall*100:.2f}%")
    print(f"Test F1-Score (macro): {f1_score*100:.2f}%")
    
    print(f"\nPER-CLASS PRECISION ({dataset_name}):")
    for i, p in enumerate(precision_per_class):
        print(f"  Digit {i}: {p*100:.2f}%")
    
    print(f"\nPER-CLASS RECALL ({dataset_name}):")
    for i, r in enumerate(recall_per_class):
        print(f"  Digit {i}: {r*100:.2f}%")
    
    # Log metrics to wandb with dataset prefix
    dataset_prefix = dataset_clean.replace('_', '_')
    wandb.log({
        f"{dataset_prefix}_test_loss": test_loss,
        f"{dataset_prefix}_test_accuracy": accuracy,
        f"{dataset_prefix}_test_precision_macro": precision * 100,
        f"{dataset_prefix}_test_recall_macro": recall * 100,
        f"{dataset_prefix}_test_f1_macro": f1_score * 100,
        f"{dataset_prefix}_confusion_matrix": wandb.Image(confusion_matrix_file)
    })
    
    # Log per-class metrics
    class_metrics = {}
    for i in range(10):
        class_metrics[f"{dataset_prefix}_precision_digit_{i}"] = precision_per_class[i] * 100
        class_metrics[f"{dataset_prefix}_recall_digit_{i}"] = recall_per_class[i] * 100
    wandb.log(class_metrics)
    
    plt.show()
    
    # Clean up the temporary image file after wandb upload
    try:
        if os.path.exists(confusion_matrix_file):
            os.remove(confusion_matrix_file)
    except:
        pass  # Don't worry if cleanup fails
    
    print("="*60)
    print(f"Evaluation on {dataset_name} completed successfully!")
    
    return {
        'accuracy': accuracy,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1_score': f1_score * 100,
        'precision_per_class': precision_per_class * 100,
        'recall_per_class': recall_per_class * 100
    }

# Run evaluation on selected dataset
try:
    # Log model configuration for reference
    wandb.log({
        "model_conv_channels": str(model.conv_channels),
        "model_fc_units": str(model.fc_units),
        "model_dropout_rate": model.dropout_rate,
        "model_use_batch_norm": model.use_batch_norm,
        "model_activation": model.activation_name
    })
    
    # Run evaluation
    result = evaluate_model(data_loader, dataset_name)
    
    print(f"\nFinal result for {dataset_name}:")
    print(f"Accuracy: {result['accuracy']:.2f}%")
    print(f"Precision: {result['precision']:.2f}%")
    print(f"Recall: {result['recall']:.2f}%")
    print(f"F1-Score: {result['f1_score']:.2f}%")
        
except KeyboardInterrupt:
    print("\nEvaluation interrupted by user (Ctrl+C)")
except Exception as e:
    print(f"\nError during evaluation: {e}")
    raise
finally:
    # Clean up wandb
    wandb.finish()
    print("Cleanup completed.")