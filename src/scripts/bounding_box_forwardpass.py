import os 
import sys
import dotenv

dotenv.load_dotenv()

# check that PYTORCH_CUDA_ALLOC_CONF is set in the environment
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    raise ValueError("PYTORCH_CUDA_ALLOC_CONF environment variable is not set")

sys.path.append(os.getcwd())

# from src.helpers.load_waybetter_db import (
    # load_waybetter_db,
    # get_pictures,
# )
from src.models.bounding_box_detection import BoundingBoxModule
import pandas as pd #noqa: E402
import sqlite3 #noqa: E402
import logging #noqa: E402

DATA_DIR = os.getenv("DATA_DIR") #May need to be changed for HF dataset
if DATA_DIR is None:
    raise ValueError("DATA_DIR environment variable is not set")
DB_PATH = str(DATA_DIR) + "/subset.sqlite3"
PHOTOS_DIR = DATA_DIR / "images"
logger = logging.getLogger("custom")
logging.basicConfig(filename="bounding_box_forward_pass.log", level=logging.CRITICAL)

# dataset = 
# dataset = dataset[dataset["image"] == 1]
sample_pictures = get_pictures(dataset, PHOTOS_DIR)

try:
    bounding_box_module = BoundingBoxModule(batch_size=10)
    bounding_box_module.run(sample_pictures)
except Exception as e:
    logger.critical(f"Error while running bounding box forward pass: {e}")
    raise
finally:
    print("Saving bounding boxes to database")
    bounding_boxes = []

    for picture in sample_pictures:
        bounding_box = picture.bounding_box
        if bounding_box is None:
            continue
        bounding_boxes.append(
            {
                "image_id": picture.original_path.name, #type: ignore
                "x1": bounding_box.x1,
                "x2": bounding_box.x2,
                "y1": bounding_box.y1,
                "y2": bounding_box.y2,
                "confidence": bounding_box.confidence,
            }
        )
    
    bounding_boxes_df = pd.DataFrame(bounding_boxes)

    # Dump the DataFrame to an SQLite database
    conn = sqlite3.connect("bounding_boxes.db")
    bounding_boxes_df.to_sql("bounding_boxes", conn, if_exists="replace", index=False)
    conn.close()
    print(f"Total bounding boxes: {len(bounding_boxes_df)}")