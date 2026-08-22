import sqlite3
import pandas as pd
from typing import List
from src.helpers.pydantic_models import WaybetterImage
from pathlib import Path
from typing import Optional
import os


def load_waybetter_db(database_path: Optional[str] = None) -> pd.DataFrame:
    if database_path is None:
        database_path = os.environ["DATABASE_PATH"]
    if not Path(database_path).exists():
        raise FileNotFoundError(f"Database file {database_path} not found")
    PHOTO_FOLDER = os.environ["PHOTOS_DIR"]
    conn = sqlite3.connect(database_path)
    df = pd.read_sql_query("SELECT * FROM weigh_ins;", conn)
    df["photo_id"] = df["photo"].str.split("/").str[-1]
    df["photo_path"] = PHOTO_FOLDER + "/" + df["photo"]
    df.set_index("photo_id", inplace=True)
    return df


def load_users_db(database_path: Optional[str] = None) -> pd.DataFrame:
    if database_path is None:
        database_path = os.environ["DATABASE_PATH"]
    if not Path(database_path).exists():
        raise FileNotFoundError(f"Database file {database_path} not found")
    conn = sqlite3.connect(database_path)
    df = pd.read_sql_query("SELECT * FROM users;", conn)
    return df

def load_bounding_boxes_db(database_path: Optional[str] = None) -> pd.DataFrame:
    if database_path is None:
        database_path = os.environ["BOUNDING_BOX_DB_PATH"]
    
    if not Path(database_path).exists():
        raise FileNotFoundError(f"Database file {database_path} not found")
    conn = sqlite3.connect(database_path)
    df = pd.read_sql_query("SELECT * FROM bounding_boxes;", conn)
    return df

def load_keypoints_db(database_path: Optional[str] = None) -> pd.DataFrame:
    if database_path is None:
        database_path = os.environ["KEYPOINTS_DB_PATH"]
    
    if not Path(database_path).exists():
        raise FileNotFoundError(f"Database file {database_path} not found")
    conn = sqlite3.connect(database_path)
    df = pd.read_sql_query("SELECT * FROM keypoints;", conn).pivot(index='image_id', columns='label', values=['x', 'y', 'confidence'])
    df.columns = [f"{label}-{coord}" for coord, label in df.columns]
    return df

def load_clusters_db(database_path: Optional[str] = None) -> pd.DataFrame:
    if database_path is None:
        database_path = os.environ["CLUSTERS_DB_PATH"]
    
    if not Path(database_path).exists():
        raise FileNotFoundError(f"Database file {database_path} not found")
    conn = sqlite3.connect(database_path)
    df = pd.read_sql_query("SELECT * FROM clusters;", conn)
    return df

def get_pictures(
    dataset: pd.DataFrame, photo_path: Path, n: int = 0
) -> List[WaybetterImage]:
    # Get N random samples from the dataset
    if n == 0:
        n = len(dataset)
    sample = dataset.sample(n)
    # Get the image paths
    image_paths = sample["photo"].tolist()
    # Get the full image paths
    image_paths = [photo_path / img_path for img_path in image_paths]
    images = [WaybetterImage(original_path=img_path) for img_path in image_paths]
    return images