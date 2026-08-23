#This file serves as a training file to get ready to make our own checkpoint. This file only SAVEs a checkpoint, never loads one.


#imports, models, dataset,

import torch #Pytorch
import torch.nn as nn
import numpy as np #Numerical Python
import tensorflow as tf #Tensorflow 
from dataset import BMIDataset #BMIDataset class
from model import DenseNet, SEDensenet121
from datasets import load_dataset

#Connect to our dataset
ds = load_dataset("Dongrae/celeb-fbi")
df_train = ds['train']
df_test = ds['test']

weight1 = SEDensenet121

batch_size = 16 #32 batches of dataset each time
epochs = 15

#Constants for image preprocessing
IMG_SIZE = 224
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]


model = load_pretrained_densenet(DenseNet) #Load our model
print(list(model.parameters()))
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001) #Load our optimization for the model

#Training loop declaration 

for epoch in range(epochs): #For our epochs we begin to utilize forward
    epoch_loss = 0.0 #Tracking an epoch loss to stop
    correct = 0
    for i in range(0, df_train, batch_size): #From 0 to the end of our data set for train we iterate each batch size
        idx = perm[i:i+batch_size] #Get the batch
        x,y = df_train[idx], df_test[idx]
        out = model(x) #Forward pass function?
        #Loss function
        loss = F.cross_entropy(out, y)
        loss.backward()
        #Optimizer
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()
        correct += (out.argmax(1) ==y).sum().item()
        n_batches = (60000 + batch_size - 1) // batch_size
        print(f"Epoch {epoch+1}/10 | Loss: {epoch_loss/n_batches:.4f} | Acc: {correct/60000*100:.1f}%")

mape = ((BMIDataset(df_train).bmi - BMIDataset(df_test).bmi)).abs() / BMIDataset(df_train).mean() * 100
print("MAPE (pandas):", round(mape, 2), "%")