"""
Scripts to evaluate performace of the model on other, unseen datasets.

Datasets tested:
- Visual Body to BMI dataset
( Reference )
- Celeb FBI
( Reference )
- VIP Attribute
( Reference )
"""

import pandas as pd
import re
import os

from src.scripts.densenet_forwardpass import forward_pass
from src.models.densenet.densenet_dataloader import WaybetterDataset
import torch
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error


def load_vip_attribute(db_file: str, photo_dir: str) -> pd.DataFrame:
    """
    Load the VIP Attribute dataset.

    Args:
    db_file (str): Path to the database file (CSV).
    photo_dir (str): Path to the directory containing the photos.
    """
    vip_attribute_df = pd.read_csv(db_file)
    vip_attribute_df["photo_path"] = vip_attribute_df["image"].apply(
        lambda x: os.path.join(photo_dir, x) + ".jpg"
    )
    vip_attribute_df.rename(columns={"BMI": "bmi", "image": "id"}, inplace=True)
    return vip_attribute_df


def load_visual_body_to_bmi(root_folder: str) -> pd.DataFrame:
    """
    Iterates over all individual folders in root_folder, parses filenames,
    and returns a pandas DataFrame with columns:
    [ 'file_path', 'individual', 'weight_lb', 'height_in', 'bmi', 'gender' ].
    """
    data = []

    # Regex pattern for parsing filenames like:
    #    1a9089_a3eWh9O_138_69_false.jpg
    #    ^------^ ^-------^ ^---^ ^^ ^---^
    #    group1   ignore    w     h  bool
    pattern = (
        r"^(?P<individual>[^_]+)"  # e.g. '2o1yfh'
        r"_[^_]+"  # e.g. '_jCcTj0Y'
        r"_(?P<weight>\d+)"  # e.g. '_231'
        r"_(?P<height>\d+)"  # e.g. '_68'
        r"_(?P<gender>true|false)"  # e.g. '_true' or '_false'
        r"(?:_.*)?\.jpg$"  # optionally capture extra text after underscore
    )

    # Iterate over each folder (one folder per individual).
    for individual_folder in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, individual_folder)

        # Make sure we only process directories (i.e., skip files in root_folder)
        if not os.path.isdir(folder_path):
            continue

        # Go through all images in this individual's folder
        for filename in os.listdir(folder_path):

            # Only consider JPG files
            if not filename.lower().endswith(".jpg"):
                continue

            # Full (absolute) path to the image
            file_path = os.path.join(folder_path, filename)

            # Use regex to extract metadata from the filename
            match = re.match(pattern, filename)
            if match:
                individual = match.group("individual")
                weight_lb = float(match.group("weight"))
                height_in = float(match.group("height"))
                gender_str = match.group("gender")

                # Convert the 'gender' string to a more readable format
                # e.g., 'true' => 'female', 'false' => 'male'
                gender = "female" if gender_str == "true" else "male"

                # Calculate BMI
                bmi = 703.0 * weight_lb / (height_in**2)

                data.append(
                    [filename, file_path, individual, weight_lb, height_in, bmi, gender]
                )

    # Build the DataFrame
    df = pd.DataFrame(
        data,
        columns=[
            "id",
            "file_path",
            "individual",
            "weight_lb",
            "height_in",
            "bmi",
            "gender",
        ],
    )

    return df


def load_celeb_fbi(root_folder):
    """
    Iterates over all images in root_folder (no subfolders), parses filenames,
    and returns a pandas DataFrame with columns:
       [ 'file_path', 'serial_no', 'height', 'weight_kg', 'gender',
         'age', 'bmi' ]
    """
    data = []

    def parse_height_string(height_str):
        """
        Given a string like '5.5', interpret it as
        5 ft 5 in. If it is '5.10', interpret as 5 ft 10 in, etc.
        Returns height in meters.
        """
        # Example: "5.5" -> 5 ft, 5 in
        #          "5.10" -> 5 ft, 10 in, etc.
        if "." not in height_str:
            # Fallback if there's no decimal (rare or malformed).
            # Could assume all is feet or return None
            return None

        parts = height_str.split(".")
        try:
            feet = int(parts[0])
            inches = int(parts[1])  # interpret the decimal part as whole inches
        except ValueError:
            return None  # can't parse properly

        total_inches = feet * 12 + inches
        height_m = total_inches * 0.0254  # convert inches to meters
        return height_m

    # Regex pattern to capture all parts of the filename.
    # Example: 1021_5.5h_51w_female_26a.png
    pattern = re.compile(
        r"^(?P<serial_no>\d+)"  # e.g. 1021
        r"_(?P<height>[0-9]+\.[0-9]+)h"  # e.g. 5.5h
        r"_(?P<weight>\d+)w"  # e.g. 51w
        r"_(?P<gender>male|female)"  # e.g. female
        r"_(?P<age>\d+)a"  # e.g. 26a
        r"\.(?:jpg|png)$",  # extension
        re.IGNORECASE,
    )

    for filename in os.listdir(root_folder):
        # Only consider .png or .jpg files
        if not (filename.lower().endswith(".png") or filename.lower().endswith(".jpg")):
            continue

        file_path = os.path.join(root_folder, filename)
        match = pattern.match(filename)
        if match:
            serial_no = match.group("serial_no")
            height_str = match.group("height")  # e.g. "5.5"
            weight_kg = float(match.group("weight"))
            gender = match.group("gender").lower()  # "male" or "female"
            age = int(match.group("age"))

            # Convert the "5.5" => 5 ft 5 in => total inches => meters
            height_m = parse_height_string(height_str)
            if not height_m:
                # Could skip or handle differently if unable to parse
                continue

            # Calculate BMI = kg / (m^2)
            bmi = weight_kg / (height_m**2)

            data.append(
                [
                    file_path,
                    serial_no,
                    f"{height_str}h",  # keep the original string plus 'h' for reference
                    weight_kg,
                    gender,
                    age,
                    bmi,
                ]
            )

    columns = ["file_path", "id", "height", "weight_kg", "gender", "age", "bmi"]
    df = pd.DataFrame(data, columns=columns)
    return df


def run_forward_pass(
    dataset: pd.DataFrame,
    checkpoint_path: str,
    output_csv: str | None = None,
    device: str = "mps",
    photo_path_col: str = "photo_path",
) -> pd.DataFrame:
    """
    dataset (pd.DataFrame): The dataset to evaluate the model on.
    checkpoint_path (str): Path to the model weights.
    device (str): Device to run the model on.
    """
    # Split the dataset into train, validation and test sets
    test_dataset = WaybetterDataset(dataset, absolute_path_col=photo_path_col)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=True, num_workers=1
    )

    trainer = forward_pass(
        checkpoint_path=checkpoint_path,
        dataloader=test_loader,
        large_model=True,
        device=device,
    )
    # Write the results to a CSV file
    results = trainer.output_df
    if output_csv:
        results.to_csv(output_csv, index=False)

    mape = mean_absolute_percentage_error(results["output"], results["target"])
    mae = mean_absolute_error(results["output"], results["target"])
    print(f"MAE: {mae}, MAPE: {mape}")

    return results