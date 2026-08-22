import os
import pandas as pd
import re

def calculate_bmi(weight_lb: float, height_in: float) -> float:
    """Calculates BMI from weight in pounds and height in inches."""
    if height_in == 0:
        return 0.0
    return (weight_lb / (height_in ** 2)) * 703

def parse_visual_bmi_dataset(base_dir: str) -> pd.DataFrame:
    """
    Parses the Visual-body-to-BMI dataset structure into a pandas DataFrame.

    The filename format is expected to be:
    individualID_imageID_weightLB_heightIN_gender.jpg
    e.g., "1a9089_a3eWh9O_138_69_false.jpg"

    Args:
        base_dir: The root directory of the dataset 
                  (e.g., 'data/visual_body_to_BMI/bodyface_1to17').

    Returns:
        A pandas DataFrame with columns:
        - individual_id (str)
        - image_id (str)
        - weight_lb (float)
        - height_in (float)
        - is_female (bool)
        - image_path (str)
        - bmi (float)
    """
    data_records = []
    
    # Regex to parse the filename
    # Example: "1a9089_a3eWh9O_138_69_false.jpg"
    #          <ind_id>_<img_id>_<w>_<h>_<g>.jpg
    # Sometimes the imageID has an extra _ in it, e.g. 10q8fn_0mII1_230_67_true_72_0.53.jpg
    # The additional numbers at the end (e.g. _72_0.53) are not part of the core spec,
    # so we make the image_id part more flexible and specifically capture the last 3 numeric/boolean parts.
    filename_pattern = re.compile(
        r"^(?P<individual_id>[^_]+)_"  # Individual ID
        r"(?P<image_id>.+)_"           # Image ID (can contain underscores)
        r"(?P<weight_lb>\d+)_"         # Weight in lbs
        r"(?P<height_in>\d+)_"         # Height in inches
        r"(?P<gender_str>true|false)"  # Gender string
        r"(?:_.*)?\.jpg$"              # Optional extra parts and .jpg extension
    )

    print(f"Scanning directory: {base_dir}")
    for root, _, files in os.walk(base_dir):
        for filename in files:
            if filename.lower().endswith('.jpg'):
                match = filename_pattern.match(filename)
                if match:
                    data = match.groupdict()
                    try:
                        individual_id = data['individual_id']
                        image_id = data['image_id']
                        weight_lb = float(data['weight_lb'])
                        height_in = float(data['height_in'])
                        is_female = data['gender_str'].lower() == 'true'
                        
                        # Validate that the parsed individual ID matches the parent directory name
                        parent_dir_name = os.path.basename(root)
                        if individual_id != parent_dir_name:
                            print(f"Warning: Mismatch between filename individual ID ('{individual_id}') and directory ('{parent_dir_name}') for {filename}. Using filename ID.")

                        image_path = os.path.join(root, filename)
                        bmi = calculate_bmi(weight_lb, height_in)

                        data_records.append({
                            'individual_id': individual_id,
                            'image_id': image_id,
                            'weight_lb': weight_lb,
                            'height_in': height_in,
                            'is_female': is_female,
                            'image_path': image_path,
                            'bmi': bmi
                        })
                    except ValueError as e:
                        print(f"Skipping file {filename} due to parsing error: {e}")
                    except Exception as e:
                        print(f"An unexpected error occurred with file {filename}: {e}")

                else:
                    # Check if it's a common system file like .DS_Store, if so, ignore silently.
                    if filename not in ['.DS_Store']:
                         print(f"Skipping file with unexpected name format: {os.path.join(root, filename)}")
                         
    if not data_records:
        print(f"No JPG files matching the expected pattern found in {base_dir} or its subdirectories.")

    return pd.DataFrame(data_records)

if __name__ == '__main__':
    # --- Configuration ---
    # Adjust this path to the root of your Visual-body-to-BMI dataset
    # It should be the directory containing the individual ID subfolders (e.g., '10q8fn', '11fex8')
    dataset_base_directory = "data/visual_body_to_BMI/bodyface_1to17" 
    output_csv_file = "data/parsed_visual_bmi_dataset.csv"
    # --- End Configuration ---

    print("Starting dataset parsing...")
    
    if not os.path.isdir(dataset_base_directory):
        print(f"Error: Dataset directory not found: {dataset_base_directory}")
        print("Please ensure 'dataset_base_directory' points to the correct location.")
    else:
        df_visual_bmi = parse_visual_bmi_dataset(dataset_base_directory)

        if not df_visual_bmi.empty:
            print(f"Successfully parsed {len(df_visual_bmi)} images.")
            print("\nFirst 5 records:")
            print(df_visual_bmi.head())
            
            print(f"\nSaving DataFrame to {output_csv_file}...")
            os.makedirs(os.path.dirname(output_csv_file), exist_ok=True)
            df_visual_bmi.to_csv(output_csv_file, index=False)
            print("DataFrame saved.")
            
            # For your finetune_densenet.py, you might need specific columns.
            # If it expects an 'absolute_path_col' and a target like 'BMI_male' or 'BMI_female',
            # you might do something like this:
            # df_visual_bmi['BMI_male'] = df_visual_bmi.apply(lambda row: row['bmi'] if not row['is_female'] else 0, axis=1)
            # df_visual_bmi['BMI_female'] = df_visual_bmi.apply(lambda row: row['bmi'] if row['is_female'] else 0, axis=1)
            # The exact columns depend on how your densenet_dataloader and trainer are set up.
            # The current `finetune_densenet.py` uses a single target column, often implicitly 'BMI'.
            # The `get_dataloader` in `densenet_dataloader.py` would need to know which column to use for labels.
            # Assuming your target is 'bmi' and image path is 'image_path'.
            print("\nTo use this with finetune_densenet.py, you would typically:")
            print("1. Load this CSV in your finetuning script.")
            print("2. Ensure the 'image_path' column is used for 'absolute_path_col'.")
            print("3. Ensure your model's target is 'bmi' or adapt the DataFrame accordingly.")
            print(f"Example for loading: pd.read_csv('{output_csv_file}')")

        else:
            print("No data was parsed. Please check the dataset path and file naming conventions.")

    print("\nScript finished.") 