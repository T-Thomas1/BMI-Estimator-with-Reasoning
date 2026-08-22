from typing import List
import cv2
import torch
from torch import Tensor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.data import MetadataCatalog
from detectron2.data.transforms import ResizeShortestEdge
from tqdm import tqdm
from src.helpers.pydantic_models import WaybetterImage, Keypoint

torch.backends.cudnn.benchmark = False


class KeyPointDetectionModule:
    """Find a human bounding box in the image

    Parameters:
        - Batch size


    Args:
        - WayBetterImage[]

    """

    def __init__(self, batch_size: int = 1):
        self.batch_size = batch_size
        self.setup_model()

    def setup_model(self) -> None:
        # Initialize configuration
        # Configuration for the model
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        )
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0  # Set the threshold
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
        )
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # Build and load the model
        model = build_model(cfg)
        model.eval()
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
        model.to(cfg.MODEL.DEVICE)

        self.cfg = cfg
        self.model = model
        self.keypoint_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).keypoint_names
        self.transform = ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
            cfg.INPUT.MAX_SIZE_TEST,
        )

    def run(self, waybetter_images: List[WaybetterImage]) -> None:
        # Process images in batches
        for i in tqdm(
            range(0, len(waybetter_images), self.batch_size), desc="Processing batches"
        ):
            try:
                batch_images = waybetter_images[i : i + self.batch_size]  # noqa
                inputs = []

                for w in batch_images:
                    original_image = cv2.imread(str(w.original_path))
                    height, width = original_image.shape[:2]
                    # Apply transformations
                    image = self.transform.get_transform(original_image).apply_image(
                        original_image
                    )
                    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).to(
                        self.cfg.MODEL.DEVICE
                    )
                    inputs.append({"image": image, "height": height, "width": width})

                with torch.no_grad():
                    outputs = self.model(inputs)

                for idx, output in enumerate(outputs):
                    instances = output["instances"]
                    max_conf_idx = instances.scores.argmax().item()
                    keypoints = instances.pred_keypoints[max_conf_idx].cpu().numpy()
                    keypoint_data = zip(self.keypoint_names, keypoints)
                    keypoints = [
                        Keypoint(label=name, x=x, y=y, confidence=conf)
                        for name, (x, y, conf) in keypoint_data
                    ]
                    batch_images[idx].keypoints = keypoints

            except Exception as e:
                print(f"Error processing batch: {e}")
                continue