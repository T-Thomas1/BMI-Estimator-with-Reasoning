# digital_scale_pb2.pyi
from typing import List

class Image:
    width: int
    height: int
    data: bytes

class ImageArray:
    images: List[Image]