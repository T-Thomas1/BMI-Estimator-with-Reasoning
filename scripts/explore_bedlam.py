# scripts/explore_bedlam.py
import os
from pathlib import Path
import pandas as pd
import json

def explore_bedlam():
    bedlam_path = Path("D:/BMI-Dataset/be_imagedata_download")

    print(f"Exploring: {bedlam_path}")
    print(f"Exists: {bedlam_path.exists()}")

    # Find and read a sample CSV file
    csv_files = list(bedlam_path.glob("**/be_seq.csv"))
    print(f"\n📊 Found {len(csv_files)} CSV files")

    if csv_files:
        # Read the first CSV to see structure
        sample_csv = csv_files[0]
        print(f"\n📄 Sample CSV: {sample_csv.parent.name}/be_seq.csv")

        df = pd.read_csv(sample_csv)
        print(f"\nColumns in CSV: {df.columns.tolist()}")
        print(f"\nFirst 5 rows:")
        print(df.head())
        print(f"\nShape: {df.shape}")

        # Check for measurement columns
        measurement_cols = [col for col in df.columns if any(x in col.lower() for x in
                                                             ['bmi', 'weight', 'height', 'shoulder', 'hip', 'waist'])]
        if measurement_cols:
            print(f"\n📏 Measurement columns found: {measurement_cols}")
            print(df[measurement_cols].head())

    # Also check ground_truth folders
    gt_folders = list(bedlam_path.glob("**/ground_truth"))
    print(f"\n📁 Found {len(gt_folders)} ground_truth folders")

    if gt_folders:
        gt_files = list(gt_folders[0].glob("*"))
        print(f"Sample ground_truth files: {[f.name for f in gt_files[:5]]}")

if __name__ == "__main__":
    explore_bedlam()