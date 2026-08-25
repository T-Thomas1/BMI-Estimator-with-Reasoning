from pydantic import BaseModel, model_validator, ConfigDict, field_validator
from typing import Optional, List, Dict, Any, Self
import pandas as pd
from PIL import Image
import os
import json
from tqdm import tqdm

OUTPUT_FILE_PREFIX = "torso_cropped_data_"

class CropMargins(BaseModel):
    margin_top_factor: float 
    margin_bottom_factor: float 
    margin_width_factor: float

    @field_validator('*')
    @classmethod
    def check_margin_factors(cls, v):
        if v < 1:
            print("Warning: Margin factors should be greater than 1. Thresholding to 1.")
        return max(1, v)

class KeypointData(BaseModel):
    left_shoulder_x: float
    left_shoulder_y: float
    right_shoulder_x: float
    right_shoulder_y: float
    left_eye_y: float
    right_eye_y: float

DEFAULT_MARGIN_FACTORS = CropMargins(
    margin_top_factor=1.5, 
    margin_bottom_factor=1.5, 
    margin_width_factor=1.5
)

class CroppingModule(BaseModel):
    data: pd.DataFrame
    keypoint_data: Optional[pd.DataFrame] = None
    margin_factors: CropMargins = DEFAULT_MARGIN_FACTORS

    image_path_col: str = 'photo_path'
    image_id_col: str = 'id'
    keypoint_cols: List[str] = [
        'left_shoulder-x', 'left_shoulder-y', 
        'right_shoulder-x', 'right_shoulder-y', 
        'left_eye-y', 'right_eye-y'
    ]

    output_dir: str = os.environ.get("CROPPED_IMAGE_DIR", "./cropped_images")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode='after')
    def check_input_data(self) -> Self:
        """
        Check if keypoint data is present in data, if not, join keypoint_data.
        Ensure all required columns are present.
        """
        starting_length = len(self.data)

        # Ensure image_path_col is in data
        if self.image_path_col not in self.data.columns:
            raise ValueError(f"Column {self.image_path_col} not found in data")

        # Ensure keypoint columns are in data
        missing_keypoint_cols = [col for col in self.keypoint_cols if col not in self.data.columns]
        if missing_keypoint_cols:
            # Try to join keypoint_data
            if self.keypoint_data is not None:
                self.data = self.data.join(
                    self.keypoint_data[self.keypoint_cols], 
                    how='left'
                )
                missing_keypoint_cols = [col for col in self.keypoint_cols if col not in self.data.columns]
                if missing_keypoint_cols:
                    raise ValueError(f"Missing keypoint columns after merging: {missing_keypoint_cols}")
            else:
                raise ValueError(
                    f"Missing keypoint columns: {missing_keypoint_cols} and keypoint_data is None"
                )

        # Check for missing values in keypoint columns
        for col in self.keypoint_cols:
            if self.data[col].isna().sum() > 0:
                raise ValueError(f"Missing values in column {col}")

        assert len(self.data) == starting_length, "Data was modified during validation!"

        return self

    def get_crop_bounding_box(self, keypoint_row: pd.Series) -> tuple:
        keypoint_data : Dict[str, Any] = keypoint_row[self.keypoint_cols].to_dict()
        keypoint_data = {key.replace('-', '_'): value for key, value in keypoint_data.items()}
        keypoints = KeypointData(**keypoint_data)
        shoulder_width = abs(keypoints.right_shoulder_x - keypoints.left_shoulder_x)

        margin_width = shoulder_width * (self.margin_factors.margin_width_factor - 1)
        margin_top = shoulder_width * (self.margin_factors.margin_top_factor - 1)
        margin_bottom = shoulder_width * (self.margin_factors.margin_bottom_factor - 1)

        crop_x_min = min(keypoints.left_shoulder_x, keypoints.right_shoulder_x) - margin_width
        crop_x_max = max(keypoints.left_shoulder_x, keypoints.right_shoulder_x) + margin_width

        crop_y_min = min(keypoints.left_eye_y, keypoints.right_eye_y) - margin_top
        crop_y_max = max(keypoints.left_shoulder_y, keypoints.right_shoulder_y) + margin_bottom

        # Ensure bounding box coordinates are valid
        return (crop_x_min, crop_y_min, crop_x_max, crop_y_max)

    def crop_and_save_image(self, image_path: str, bounding_box: tuple, output_path: str) -> None:
        with Image.open(image_path) as img:
            # Ensure bounding box is within image dimensions
            img_width, img_height = img.size
            crop_x_min, crop_y_min, crop_x_max, crop_y_max = bounding_box

            crop_x_min = max(0, crop_x_min)
            crop_y_min = max(0, crop_y_min)
            crop_x_max = min(img_width, crop_x_max)
            crop_y_max = min(img_height, crop_y_max)

            # Crop and save image
            cropped_img = img.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cropped_img.save(output_path)

    @property
    def cropped_data(self) -> pd.DataFrame:
        """
        Apply cropping to all images, save cropped images, and return updated DataFrame
        """
        # Create a copy of data to avoid modifying original
        data = self.data.copy()

        # Prepare list to store paths to cropped images
        cropped_image_paths = []

        for idx, row in tqdm(data.iterrows(), total=len(data)):
            try:
                image_path = row[self.image_path_col]
                # Compute bounding box
                bounding_box = self.get_crop_bounding_box(row)
                # Define output path
                filename = os.path.basename(image_path)
                output_path = os.path.join(self.output_dir, filename)
                # Crop and save image
                self.crop_and_save_image(image_path, bounding_box, output_path)
                # Append output path
                cropped_image_paths.append(output_path)
            except Exception as e:
                print(f"Error processing image {row[self.image_id_col]}: {e}")
                cropped_image_paths.append(None)

        # Add new column to data
        data['cropped_image_path'] = cropped_image_paths

        print(f"Cropped images saved to {self.output_dir}")
        return data

    @property
    def __hash__(self) -> int:
        """Generate hash based on margin factors and data length"""
        return hash((
            self.margin_factors.margin_top_factor,
            self.margin_factors.margin_bottom_factor,
            self.margin_factors.margin_width_factor,
            len(self.data)
        ))

    def save_cropped_data(self, path: str = os.environ.get("CUSTOM_DATA_DIR", "./")) -> str:
        """Save updated data to a file with hash as filename"""
        filename = f"{OUTPUT_FILE_PREFIX}{self.__hash__}.csv"

        full_path = os.path.join(path, filename)
        if filename in os.listdir(path):
            raise FileExistsError(f"File {filename} already exists in {path}")
        output_data = self.cropped_data
        output_data.to_csv(full_path, index=False)
        print(f"Cropped data saved to {full_path}")

        # Save settings to a JSON file
        settings = {
            "margin_top_factor": self.margin_factors.margin_top_factor,
            "margin_bottom_factor": self.margin_factors.margin_bottom_factor,
            "margin_width_factor": self.margin_factors.margin_width_factor
        }

        settings_file = os.path.join(path, "cropping_settings.json")
        with open(settings_file, "a") as f:
            f.write(json.dumps({self.__hash__: settings}) + "\n")

        return full_path