import os
import sys
import dotenv
sys.path.append(os.getcwd())
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


from src.helpers.load_waybetter_db import load_waybetter_db
from src.models.densenet.densenet_dataloader import get_dataloader
from src.models.densenet.densenet_trainer import Trainer
from src.models.densenet import densenet
from src.helpers.split_dataset import split_dataframe_without_user_overlap
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional

dotenv.load_dotenv()
print(f"Photo path: {os.environ.get('PHOTOS_DIR')}")


def train_densenet(waybetter_data: pd.DataFrame | str = "filtered_dataset.db",
                   save_dir: str = "save_dir",
                   epochs: int = 20,
                   batch_size: int = 32,
                   absolute_path_col: Optional[str] = None,
                   large_model: bool = False):
    if isinstance(waybetter_data, str):
        waybetter_data = load_waybetter_db(database_path=waybetter_data)

    waybetter_df = split_dataframe_without_user_overlap(waybetter_data)
    train_loader, val_loader, test_loader = get_dataloader(waybetter_df, batch_size=batch_size, num_workers=4, absolute_path_col=absolute_path_col)

    if large_model:
        model = densenet.model_large
        densenet.load_pretrained_densenet201(model)
    else:
        model = densenet.model
        densenet.load_pretrained_densenet(model)

    DEVICE = torch.device("cuda")
    LR = 0.0001
    WEIGHT_DECAY = 0.0001

    criterion = nn.MSELoss().to(DEVICE)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR,
                            weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    trainer = Trainer(model, DEVICE, optimizer, criterion, save_dir=save_dir)
    trainer.Loop(epochs, train_loader, val_loader, scheduler)

# CROPPED DATA TRAINER
# cropped_data = pd.read_csv('/home/rajiv/rajiv-old/DigitalScale/data/filtered_datasets/filtered_data_december_with_torso_image.csv')
# cropped_data = cropped_data.dropna(subset=['cropped_image_path'])
# # Drop original image path column to be sure we're using the cropped image path
# cropped_data = cropped_data.drop(columns=['photo'])
# train_densenet(cropped_data, save_dir='trained_models/torso_only_large', batch_size=64, epochs=40, absolute_path_col='cropped_image_path', large_model=True)

# FACE DATA TRAINER
face_data = pd.read_csv('/home/rajiv/rajiv-old/DigitalScale/data/filtered_datasets/filtered_data_december_with_face_only.csv')
face_data = face_data.dropna(subset=['face_only_photo_path'])
# Drop original image path column to be sure we're using the cropped image path
face_data = face_data.drop(columns=['photo'])
train_densenet(face_data, save_dir='trained_models/face_only_large', batch_size=64, epochs=40, absolute_path_col='face_only_photo_path', large_model=True)

# FULL DATA TRAINER
#train_densenet("data/filtered_datasets/filtered_data_december.db", save_dir='trained_models/no_user_overlap_40_epochs', batch_size=64, epochs=40)

## Large model trainer
# train_densenet("data/filtered_datasets/filtered_data_december.db", save_dir='trained_models/no_user_overlap_40_epochs_large', batch_size=64, epochs=40, large_model=True)