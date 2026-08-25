#Conformal Prediction for predict_bmi.py to provide a split conformal regression value. 

import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from datasets import load_dataset
import sys
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
import scipy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import argparse
from model import SEDensenet121, SEDensenet201, load_pretrained_densenet #, load_pretrained_densenet201
from dataset import BMIDataset, df_test, df_train
from predict_bmi import load_model, predict_bmi
from mapie.regression import MapieRegressor
from mapie.conformity_scores import AbsoluteConformityScore

parser = argparse.ArgumentParser(description='BMI Prediction using DenseNet')
parser.add_argument('--model_path', type=str, default='weights/best_model.ckpt',
                    help='Path to model checkpoint')
parser.add_argument('--model_type', type=str, default='densenet121',
                    choices=['densenet121', 'densenet201'],
                    help='Type of model to use')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size for inference')
parser.add_argument('--device', type=str, default='auto',
                    choices=['auto', 'cuda', 'cpu'],
                    help='Device to use for computation')
# Set device
if args.device == 'auto':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
else: 
    device = args.device
args = parser.parse_args()
#Load our dataset
test_dataset = BMIDataset(df_test)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

calibration_set = Subset(test_dataset, range(0, 350)) #Calibration set
evaluation_set = Subset(test_dataset, range (350, len(test_dataset))) #Evaluation set

def calibrate_bmi(model, dataloader, device="cuda"):
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for images, bmis, ids in test_dataloader:
            images = images.to(device, dtype=torch.float32)
            preds = model(images).cpu().numpy().flatten()
            for p, y in zip(preds, bmis):
                y_hat.append(float(p))
                y_true.append(y.item())
    y_true = np.array(y_true)
    y_hat = np.array(y_hat)
    scores = np.abs(y_true - y_hat) #Split conformal prediction
    n = len(scores)
    alpha = 0.1
    k = int(np.ceil((n + 1) * (1 - alpha)))
    q_hat = float(np.sort(scores)[k-1])
    
model = load_model(args.model_path, args.model_type, device) #Load our model
calibrate_bmi(model, test_dataloder, device)