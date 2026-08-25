#Conformal Prediction for predict_bmi.py to provide a prediction with interval

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
from torch.utils.data import DataLoader
import argparse
from model import SEDensenet121, SEDensenet201, load_pretrained_densenet #, load_pretrained_densenet201
from dataset import BMIDataset, df_test, df_train
from predict_bmi import load_model, predict_bmi

#Load our dataset
sample_dataset = BMIDataset(df_test)
sample_dataloader = DataLoader(sample_dataset, batch_size=args.batch_size, shuffle=False)
true_dataset = BMIDataset(df_train)
true_dataloader = DataLoader(true_dataset, batch_size=args.batch_size, shuffle=False)

#Load model 
print("Loading model...")
model = load_model(args.model_path, args.model_type, device)
if model is None:
    print("Failed to load model. Exiting...")
    sys.exit(1)

x_calib = sample_dataset[3000:] #Calibrations
y_calib = true_dataset[350:]
calib_dataloader = DataLoader(y_calib, batch_size=args.batch_size, shuffle=False)
x_eval = sample_dataset[:3000] #Evaluations
y_eval = true_dataset[:350]

alpha = 0.1 # 10% error/not right 
if args.device == 'auto':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
else: 
    device = args.device

predict_bmi(calib_dataloader, true_dataloader, device) #Calibrate

def calibrate(self, calibrated_dataloader, alpha = 0.1):

    if self._model is None:
        raise ValueError("Model is not defined. Please provide a valid model.")
    
    self._model.eval()
    logits_list = [] #Logits to keep track of strengh
    labels_list = [] #For what the model labels
    with torch.no_grad():
        for examples in calibrated_dataloader:
            tmp_x, tmp_labels = examples[0].to(self._device), examples[1].to(self._device)
            tmp_logits = self._logits_transformation(self._model(tmp_x)).detach()
            logits_list.append(tmp_logits)
            labels_list.append(tmp_labels)
        logits = torch.cat(logits_list).float()
        labels = torch.cat(labels_list)
    self.calculate_threshold(logits, labels, alpha)

def calculate_threshold(self, logits, labels, alpha = 0.1):
    logits = logits.to(self._device)
    labels = labels.to(self._device)
    self.cal_scores = self.score_function(logits, labels)
    self.q_hat = self._calculate_conformal_value(self.cal_scores, alpha)

def _calculate_conformal_value(self, scores, alpha):
    return calculate_conformal_value(scores, alpha)