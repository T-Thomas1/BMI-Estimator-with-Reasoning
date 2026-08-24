import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from datasets import load_dataset

ds = load_dataset("Dongrae/celeb-fbi")
df_train = ds['train']
df_test = ds['test']

#Constants for image preprocessing
IMG_SIZE = 224
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

class CustomResize:
    """ Custom resize transform that maintains aspect ratio"""
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            h, w = img.shape[-2:]
        elif isinstance(img, Image.Image):
            w, h = img.size
        else:
            h, w = img.shape[:2]

        scale = max(w, h) / float(self.size)
        new_w, new_h = int(w / scale), int(h / scale)
        return transforms.functional.resize(img, (new_h, new_w))

class BMIDataset(Dataset):
    """Dataset class for BMI prediction from images"""
    def __init__(self, hf_dataset, image_col='image'):
        valid_indices = []
        for idx in range(len(hf_dataset)):
            row = hf_dataset[idx]
            weight = row.get('weight', None) #Additional metadata for calculation of BMI
            height = row.get('height', None)
            if(weight is not None and height is not None and weight != -1 and height != -1):
                valid_indices.append(idx)
        self.df = hf_dataset.select(valid_indices)
        self.image_col = image_col
        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB') if hasattr(img, 'mode') and img.mode != 'RGB' else img),
            CustomResize(IMG_SIZE),
            transforms.Pad(IMG_SIZE),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMG_MEAN, IMG_STD)
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df[idx]

        #Load and preprocess image
        image = row[self.image_col]

        #Get BMI value (if available) - Not applicable in our dataset, but kept for compatibility with original dataset structure
        # bmi = row.get('bmi', None)

        if isinstance(image, dict):

            if 'image' in image:
                image = image['image']
            elif 'array' in image:
                image = image['array']
            else:

                for key, value in image.items():
                    if hasattr(value, 'size') or hasattr(value, 'shape'):
                        image = value
                        break
                    
        image = self.transform(image)
        weight = row.get('weight', None) #Additional metadata for calculation of BMI
        height = row.get('height', None)
        bmi = weight / (height / 100.0) ** 2
        return image, bmi, row.get('id', idx)

    def load_sample_data(load_dataset="data"):
        """ Load sample data for BMI Prediction.
        
        Args:
            load_dataset: Directory containing the dataset
            
        Returns:
            DataFrame with image paths and metadata
        """

        csv_path = os.path.join(load_dataset, "Dongrae/celeb-fbi")

        if not os.path.exists(csv_path):
            print(f"Dataset CSV not found at {csv_path}.")
            print("Please download the sample data from the authors: ")
            print("Contact: rmanichand@ethz.ch or planger@ethz.ch")
            return None

        #Load the full dataset
        df = pd.read_csv(csv_path)

        #Get list of available folders
        available_folders = [d for d in os.listdir(load_dataset)
                             if os.path.isdir(os.path.join(load_dataset, d))]

        #Filter dataframe to only include images from available folders
        df['folder'] = df['individual_id'].astype(str)
        sample_data = df[df['folder'].isin(available_folders)].copy()

        #Updated image paths to match our local structure
        sample_data['image'] = sample_data.apply(
            lambda row: os.path.join(load_dataset, row['folder'], os.path.basename(row['image'])),
            axis=1
            )

        # Verify images exist and keep only those that do
        sample_data = sample_data[sample_data['image'].apply(os.path.exists).reset_index (drop=True)]

        if len(sample_data) == 0:
            print("No valid images found. Please check the data directory structure.")
            print("Expected structure:")
            print("  data/")
            print("    visual-body-to-bmi.csv")
            print("    individual_id_folders/")
            print("      *.jpg or *.png files")
            return None

        print(f"Found {len(sample_data)} valid images for testing")
        return sample_data
