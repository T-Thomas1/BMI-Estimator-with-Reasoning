import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src"))

import dotenv
from src.helpers.load_waybetter_db import load_waybetter_db
from retinaface import RetinaFace
import numpy as np
from tqdm import tqdm

dotenv.load_dotenv()

OUTPUT_CSV = "data/face_bounding_boxes.csv"

def find_face_bounding_box(waybetter_db, output_csv):
    # Initialize new columns
    waybetter_db["x_min"] = np.nan
    waybetter_db["y_min"] = np.nan
    waybetter_db["x_max"] = np.nan
    waybetter_db["y_max"] = np.nan
    waybetter_db["face_confidence_score"] = np.nan

    for i in tqdm(range(len(waybetter_db))):
        photo_path = waybetter_db.iloc[i]["photo_path"]
        try:
            resp = RetinaFace.detect_faces(photo_path)

            if len(resp) == 0:
                print(f"Warning: No faces detected for index {i}")
            else:
                # If multiple faces, pick the one with the highest score
                best_face = max(resp.values(), key=lambda f: f["score"])
                x_min, y_min, x_max, y_max = best_face["facial_area"]
                waybetter_db.iloc[i, waybetter_db.columns.get_loc("x_min")] = x_min
                waybetter_db.iloc[i, waybetter_db.columns.get_loc("y_min")] = y_min
                waybetter_db.iloc[i, waybetter_db.columns.get_loc("x_max")] = x_max
                waybetter_db.iloc[i, waybetter_db.columns.get_loc("y_max")] = y_max
                waybetter_db.iloc[i, waybetter_db.columns.get_loc("face_confidence_score")] = best_face["score"]

                if len(resp) > 1:
                    print(f"Warning: Multiple faces detected for index {i}. Storing only highest score.")
        except Exception as e:
            print(f"Warning: Could not process index {i}: {e}")

    waybetter_db.to_csv(output_csv, index=False)
    return waybetter_db

waybetter_db = load_waybetter_db()[["id", "user_id", "photo", "photo_path"]]

find_face_bounding_box(waybetter_db, OUTPUT_CSV)