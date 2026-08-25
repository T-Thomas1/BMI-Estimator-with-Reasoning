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
from torch.utils.data import DataLoader, Subset
import argparse
from model import SEDensenet121, SEDensenet201, load_pretrained_densenet #, load_pretrained_densenet201
from dataset import BMIDataset, df_test, df_train
from predict_bmi import load_model, predict_bmi
from mapie.regression import MapieRegressor
from mapie.conformity_scores import AbsoluteConformityScore

#Load our dataset
test_dataset = BMIDataset(df_test)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
train_dataset = BMIDataset(df_train)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
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

#Load model 
print("Loading model...")
model = load_model(args.model_path, args.model_type, device)
if model is None:
    print("Failed to load model. Exiting...")
    sys.exit(1)

x_train = Subset(test_dataset, range(0, 3000)) #Sample training
y_train = Subset(test_dataset, range(3000, len(test_dataset))) # Label training

x_calib = Subset(train_dataset, range(0, 350)) #Sample calibration
y_calib = Subset(train_dataset, range(350, len(train_dataset))) # Label calibration

test_dataloader = DataLoader(y_calib, batch_size=args.batch_size, shuffle=False) #Dataloader for calibration


x_eval = sample_dataset[:3000] #Evaluations
y_eval = true_dataset[:350]

alpha = 0.1 # 10% error/not right 
if args.device == 'auto':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
else: 
    device = args.device


def construct_prediction_bands(mean_prediction, quantiles):
    """
    A function that returns the lower and upper prediction bands for every quantile in quantiles.
    """
    
    prediction_bands = np.zeros((len(mean_prediction), 2, len(quantiles)))

    for i, q in enumerate(quantiles):
        prediction_bands[:, :, i] = np.stack([
            mean_prediction - q,
            mean_prediction + q
        ], axis=1)

    return prediction_bands

mean_predictor = LinearRegression()
mean_predictor.fit(X_train, y_train)
y_train_predict = mean_predictor.predict(X_train)

mapie_regressor = MapieRegressor(mean_predictor, conformity_score=None, cv = 'prefit', method ='base')
mapie_regressor.fit(x_calib, y_calib)
y_pred, y_pis = mapie_regressor.predict(x_train, 0.1)

predict_bmi(calib_dataloader, true_dataloader, device) #Calibrate

# For the mean predictor, we choose a linear regression model as this matches the ground-truth function
mean_predictor = LinearRegression()
mean_predictor.fit(x_calib, y_calib)
y_calib_predict = mean_predictor.predict(x_calib)

cal_mean_prediction = mean_predictor.predict(x_calib)

scores = np.abs(y_calib - cal_mean_prediction) #Split conformal prediction