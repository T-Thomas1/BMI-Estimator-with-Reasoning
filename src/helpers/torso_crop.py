from pydantic import BaseModel, field_validator
from PIL import Image
import pandas as pd
from typing import Dict, Any

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

    

DEFAULT_MARGIN_FACTORS = CropMargins(margin_top_factor=1.5, margin_bottom_factor=1.5, margin_width_factor=1.5)

def crop_to_torso(image: Image, keypoint_data: pd.Series | Dict[str, Any], margin_factors: CropMargins = DEFAULT_MARGIN_FACTORS) -> Image:
    # Verify that the keypoint data is in the correct format by setting it to the pydantic model

    if isinstance(keypoint_data, pd.Series):
        keypoint_data = keypoint_data.to_dict()
        keypoint_data = {key.replace('-', '_'): value for key, value in keypoint_data.items()}

    keypoints = KeypointData(**keypoint_data)
    shoulder_width = abs(keypoints.right_shoulder_x - keypoints.left_shoulder_x)

    margin_width = shoulder_width * (margin_factors.margin_width_factor - 1)
    margin_top = shoulder_width * (margin_factors.margin_top_factor - 1)
    margin_bottom = shoulder_width * (margin_factors.margin_bottom_factor - 1)

    crop_x_min = min(keypoints.left_shoulder_x, keypoints.right_shoulder_x) - margin_width
    crop_x_max = max(keypoints.left_shoulder_x, keypoints.right_shoulder_x) + margin_width

    crop_y_min = min(keypoints.left_eye_y, keypoints.right_eye_y) - margin_top
    crop_y_max = max(keypoints.left_shoulder_y, keypoints.right_shoulder_y) + margin_bottom

    return image.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))

def get_crop_bounding_box(keypoint_data: pd.Series | Dict[str, Any], margin_factors: CropMargins = DEFAULT_MARGIN_FACTORS) -> tuple:
    if isinstance(keypoint_data, pd.Series):
        keypoint_data = keypoint_data.to_dict()
    keypoint_data = {key.replace('-', '_'): value for key, value in keypoint_data.items()}

    keypoints = KeypointData(**keypoint_data)
    shoulder_width = abs(keypoints.right_shoulder_x - keypoints.left_shoulder_x)

    margin_width = shoulder_width * (margin_factors.margin_width_factor - 1)
    margin_top = shoulder_width * (margin_factors.margin_top_factor - 1)
    margin_bottom = shoulder_width * (margin_factors.margin_bottom_factor - 1)

    crop_x_min = min(keypoints.left_shoulder_x, keypoints.right_shoulder_x) - margin_width
    crop_x_max = max(keypoints.left_shoulder_x, keypoints.right_shoulder_x) + margin_width

    crop_y_min = min(keypoints.left_eye_y, keypoints.right_eye_y) - margin_top
    crop_y_max = max(keypoints.left_shoulder_y, keypoints.right_shoulder_y) + margin_bottom

    return (crop_x_min, crop_y_min, crop_x_max, crop_y_max)