#!/usr/bin/env python3

"""
Simple BMI prediction script using trained DenseNet models.

This script demonstrates how to load a pre-trained model and run BMI predictions
on sample images from the dataset.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse

from model import SEDensenet121, SEDensenet201, load_pretrained_densenet #, load_pretrained_densenet201
# from datasets import load_dataset
from dataset import BMIDataset, df_test, df_train

def load_model(model_path, model_type="densenet121", device="cuda"):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint file
        model_type: Type of model ("densenet121" or "densenet201")
        device: Device to load model on
        
    Returns:
        Loaded model in evaluation mode
    """
    if not os.path.exists(model_path):
        print(f"Model checkpoint not found at {model_path}")
        print("Please download the model weights from the authors")
        print("Contact: rmanichand@ethz.ch or planger@ethz.ch")
        return None
    # Initialize the model with pre-trained weights
    if model_type.lower() == "densenet121":
        model = SEDensenet121()
    elif model_type.lower() == "densenet201":
        model = SEDensenet201()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    print(f"Loaded {model_type} model from {model_path}")
    return model

def predict_bmi(model, dataloader, device="cuda"):
    """
    Run BMI prediction on a dataset.
    
    Args:
        model: Trained model
        dataloader: DataLoader with images
        device: Device for computations
        
    Returns:
        List of results with predictions and metadata
    """
    results = []
    model.eval()

    with torch.no_grad():
        for batch_idx, (images, bmis, individual_ids) in enumerate(dataloader):
            images = images.to(device, dtype=torch.float32)

            # Get predictions
            outputs = model(images)
            predicted_bmis = outputs.cpu().numpy().flatten()

            # Store results
            for i in range(len(images)):
                result = {
                    'individual_id': individual_ids[i],
                    'predicted_bmi': predicted_bmis[i],
                    'actual_bmi': bmis[i].item() if bmis[i] is not None else None,
                }
                results.append(result)

                # Print progress
                if result['actual_bmi'] is not None:
                    print(f"Individual ID: {result['individual_id']}, "
                          f"Predicted BMI: {result['predicted_bmi']:.2f}, "
                          f"Actual BMI: {result['actual_bmi']:.2f}")
                else:
                    print(f"ID: {result['individual_id']:<10} | "
                          f"Predicted: {result['predict_bmi']:.2f}")
    return results

def main():
    parser = argparse.ArgumentParser(description='BMI Prediction using DenseNet')
    parser.add_argument('--model_path', type=str, default='weights/best_model.ckpt',
                        help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='densenet121',
                        choices=['densenet121', 'densenet201'],
                        help='Type of model to use')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use for computation')

    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else: 
        device = args.device

    print(f"Using device: {device}")

    # Create dataset and dataloader
    dataset = BMIDataset(df_test)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    #Load model 
    print("Loading model...")
    model = load_model(args.model_path, args.model_type, device)
    if model is None:
        print("Failed to load model. Exiting...")
        sys.exit(1)

    # Run predictions
    print("\nRunning BMI predictions...")
    print("-" * 80)
    results = predict_bmi(model, dataloader, device)

    # Calculate summary statistics 
    if any(r['actual_bmi'] is not None for r in results):
        errors = [abs(r['predicted_bmi'] - r['actual_bmi'])
                  for r in results if r['actual_bmi'] is not None]

        print("-" * 80)
        print(f"Summary Statistics (n{len(errors)}):")
        print(f"Average Error: {sum(errors)/len(errors):.2f}")
        print(f"Max Error: {max(errors):.2f}")
        print(f"Min Error: {min(errors):.2f}")

    print(f"\nCompleted predictions for {len(results)} images.")

if __name__ == "__main__":
    main()