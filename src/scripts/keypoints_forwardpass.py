import os
import sys
import dotenv

dotenv.load_dotenv()

# check that PYTORCH_CUDA_ALLOC_CONF is set in the environment
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    raise ValueError("PYTORCH_CUDA_ALLOC_CONF environment variable is not set")

sys.path.append(os.getcwd())

from src.helpers.load_waybetter_db import (  # noqa: E402
    load_waybetter_db,
    get_pictures,
)
from src.models.keypoint_detection import KeyPointDetectionModule  # noqa: E402
import pandas as pd  # noqa: E402
import sqlite3  # noqa: E402
import logging  # noqa: E402
from pathlib import Path  # noqa: E402

WOKRING_DIR = Path("~/DigitalScale")

DATA_DIR = Path(os.getenv("DATA_DIR"))
if DATA_DIR is None:
    raise ValueError("DATA_DIR environment variable is not set")
DB_PATH = str(DATA_DIR) + "/subset.sqlite3"
PHOTOS_DIR = DATA_DIR / "images"
logger = logging.getLogger("custom")
logging.basicConfig(filename="bounding_box_forward_pass.log", level=logging.CRITICAL)


dataset = load_waybetter_db(DB_PATH)
dataset = dataset[dataset["image_present"] == 1]
sample_pictures = get_pictures(dataset, PHOTOS_DIR)


try:
    keypoint_detection_module = KeyPointDetectionModule(batch_size=16)
    keypoint_detection_module.run(sample_pictures)
except Exception as e:
    logger.critical(f"Error while running bounding box forward pass: {e}")
    raise
finally:
    print("Saving bounding boxes to database")
    all_keypoints = []

    for picture in sample_pictures:
        if picture.keypoints is None:
            continue

        for keypoint in picture.keypoints:
            all_keypoints.append(
                {
                    "image_id": picture.original_path.name,  # type: ignore
                    "x": keypoint.x,
                    "y": keypoint.y,
                    "confidence": keypoint.confidence,
                    "label": keypoint.label,
                }
            )

    keypoints_df = pd.DataFrame(all_keypoints)

    # Dump the DataFrame to an SQLite database
    conn = sqlite3.connect("keypoints.db")
    keypoints_df.to_sql("keypoints", conn, if_exists="replace", index=False)
    conn.close()
    print(f"Total keypoints: {len(keypoints_df)}")