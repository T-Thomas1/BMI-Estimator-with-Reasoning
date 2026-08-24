#This file serves as a training file to get ready to make our own checkpoint. This file only SAVEs a checkpoint, never loads one.
import os
import torch #Pytorch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np #Numerical Python
from dataset import BMIDataset #BMIDataset class
from model import DenseNet, SEDensenet121, load_pretrained_densenet
from datasets import load_dataset

#Connect to our dataset
ds = load_dataset("Dongrae/celeb-fbi")
df_train = ds['train']
df_test = ds['test']


batch_size = 16 #32 batches of dataset each time
epochs = 40 #DigitalScale utilized 40 so we will too

#Constants for image preprocessing
IMG_SIZE = 224
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]


model = SEDensenet121()
load_pretrained_densenet(model) #Load our model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

train_loader = DataLoader(BMIDataset(df_train), batch_size=16, shuffle=True, num_workers=0, pin_memory=True) #Mini batches when loading data. Shuffle, slice, yield.
val_loader = DataLoader(BMIDataset(df_test), batch_size=16, shuffle=False, num_workers=0, pin_memory=True)


optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, weight_decay=1e-4) #Load our optimization for the model
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5) #Adaptive scheduler

criterion = nn.MSELoss()
#Training loop declaration 
best_mae = float('inf')
for epoch in range(epochs): #For our epochs we begin to utilize forward
    model.train() ##Training 

    #TRAINING (LEARNING) Guess -> See how wrong -> Adjust
    for images, bmi, _ in train_loader: #For images, bmi and something else we dont focus on 'id' 
        images, bmi = images.to(device), bmi.to(device).float() #push these images/bmi to our device for training
        optimizer.zero_grad() #Remove previous gradient to prepare our training loop
        pred = model(images).squeeze(1) #remove the value
        loss = criterion(pred, bmi) #To pass to our loss
        loss.backward() ##After forward, we begin backward
        optimizer.step() #our optimizer would step after the backwards use

    model.eval() #Validation during the training loop as a security layer. 
    val_mae = val_mape = 0.0
    n = 0
    with torch.no_grad():

        # VALIDATION (MEASURING) Its a report card of sorts
        for images, bmi, _ in val_loader:
            images, bmi = images.to(device), bmi.to(device).float()
            pred = model(images).squeeze(1)
            val_mae += (pred - bmi).abs().sum().item()
            val_mape += ((pred - bmi).abs() / bmi * 100).sum().item()
            n += bmi.size(0)
    val_mae /= n
    val_mape /= n 
    scheduler.step(val_mae)
    print(f"Epoch {epoch+1}/{epochs} | val MAE {val_mae:.3f} | val MAPE {val_mape:.2f}%")

    if val_mae < best_mae:
        best_mae = val_mae
        os.makedirs('weights', exist_ok=True)
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'MAE': val_mae, 'MAPE': val_mape, 'optimizer': optimizer.state_dict(),}, 'weights/best_model.ckpt')