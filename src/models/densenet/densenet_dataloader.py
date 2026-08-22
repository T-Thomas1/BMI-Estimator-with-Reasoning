import pandas as pd
import torch
from torchvision import transforms
import os
import cv2
from typing import Optional
import src.models.densenet.utils as densenet_utils  # type: ignore


# STD and Mean from imagenet
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224


def get_dataloader(
    waybetter_dataframe: pd.DataFrame,
    batch_size: int,
    num_workers: int,
    photos_path: Optional[str] = None,
    absolute_path_col: Optional[str] = None,
):
    assert "partition" in waybetter_dataframe.columns

    train_df = waybetter_dataframe[waybetter_dataframe["partition"] == "train"].reset_index(
        drop=True
    )
    val_df = waybetter_dataframe[waybetter_dataframe["partition"] == "val"].reset_index(
        drop=True
    )
    test_df = waybetter_dataframe[waybetter_dataframe["partition"] == "test"].reset_index(
        drop=True
    )

    train_dataset = WaybetterDataset(
        waybetter_dataframe=train_df,
        photos_path=photos_path,
        absolute_path_col=absolute_path_col,
    )
    val_dataset = WaybetterDataset(
        waybetter_dataframe=val_df,
        photos_path=photos_path,
        absolute_path_col=absolute_path_col,
    )
    test_dataset = WaybetterDataset(
        waybetter_dataframe=test_df,
        photos_path=photos_path,
        absolute_path_col=absolute_path_col,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


class WaybetterDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        waybetter_dataframe: pd.DataFrame,
        photos_path: Optional[str] = None,
        absolute_path_col: Optional[
            str
        ] = None,  # Use if input data contains absolute path to image. Will overwrite photos_path
    ):

        if photos_path is None:
            self.photos_path = os.environ.get("PHOTOS_DIR")
        else:
            self.photos_path = photos_path

        self.absolute_path_col = absolute_path_col

        self.weigh_ins_df = waybetter_dataframe
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                densenet_utils.Resize(IMG_SIZE),
                transforms.Pad(IMG_SIZE),
                transforms.CenterCrop(IMG_SIZE),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.weigh_ins_df)

    def __getitem__(self, idx):
        row = self.weigh_ins_df.iloc[idx]

        if self.absolute_path_col:
            image = cv2.imread(row[self.absolute_path_col])
        else:
            image = cv2.imread(os.path.join(self.photos_path, row["photo"]))

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img = self.transform(img)
        img = transforms.Normalize(IMG_MEAN, IMG_STD)(img)

        sex, BMI = "F", row["bmi"]

        # Check if any of the values are None
        if any(v is None for v in [img, sex, BMI]):
            if img is None:
                print("Image is None")
            if sex is None:
                print("sex is None")
            if BMI is None:
                print("BMI is None")
            raise ValueError("One of the values is None")

        return (img, (sex, BMI, row["id"]))