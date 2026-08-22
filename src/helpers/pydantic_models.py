from pydantic import BaseModel
from typing import Optional
from pathlib import Path


class WaybetterImage(BaseModel):
    original_path: str | Path
    bounding_box: Optional["HumanBoundingBox"] = None
    keypoints: Optional[list["Keypoint"]] = None


class HumanBoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def size(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


class Keypoint(BaseModel):
    x: float
    y: float
    label: str
    confidence: float