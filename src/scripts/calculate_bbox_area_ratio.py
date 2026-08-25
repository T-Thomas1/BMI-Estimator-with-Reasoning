from src.helpers.load_waybetter_db import load_waybetter_db
import cv2
import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas()
import sqlite3
from os import environ


DB_PATH = environ.get("BOUNDING_BOX_DB_PATH")

PHOTO_FOLDER = environ.get("PHOTOS_DIR")


bounding_boxes = pd.read_sql(
    "SELECT * FROM bounding_boxes", sqlite3.connect(DB_PATH)
).set_index("image_id")

# For each bbox calculate the
# area compared to the image area
bounding_boxes["width"] = bounding_boxes["x2"] - bounding_boxes["x1"]
bounding_boxes["height"] = bounding_boxes["y2"] - bounding_boxes["y1"]
bounding_boxes["area"] = bounding_boxes["width"] * bounding_boxes["height"]
assert (bounding_boxes["width"] >= 0).all()
assert (bounding_boxes["height"] >= 0).all()
assert (bounding_boxes["area"] >= 0).all()

waybetter_db = load_waybetter_db()
# Create photo id column by splitting photo column on / and taking the last element
waybetter_db["photo_id"] = waybetter_db["photo"].str.split("/").str[-1]
waybetter_db.set_index("photo_id", inplace=True)


def get_photo_area(image_id: str) -> float:
    photo_path = PHOTO_FOLDER + "/" + waybetter_db.loc[image_id].photo
    photo = cv2.imread(photo_path)
    return photo.shape[0] * photo.shape[1]


bounding_boxes["photo_area"] = bounding_boxes.index.to_series().progress_map(get_photo_area)
bounding_boxes["bbox_area_ratio"] = bounding_boxes["area"] / bounding_boxes["photo_area"]