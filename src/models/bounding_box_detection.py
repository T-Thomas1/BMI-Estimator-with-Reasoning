from typing import List, Optional, Tuple, Any
from pydantic import BaseModel
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
from src.helpers.pydantic_models import WaybetterImage, HumanBoundingBox
import time

torch.backends.cudnn.benchmark = False


def preprocess_image(orig_image: Any, cfg: Any) -> Tensor:
    """
    Preprocess the image as per the model's requirements.
    """
    # Create a copy to avoid modifying the original image
    image = orig_image.copy()
    # Apply resizing transformation
    transform_gen = ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )
    image = transform_gen.get_transform(image).apply_image(image)
    # Convert to torch tensor
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    # Normalize the image
    image = (image - torch.tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)) / torch.tensor(
        cfg.MODEL.PIXEL_STD
    ).view(-1, 1, 1)
    return image


def preprocess_image_small(orig_image: Any, cfg: Any) -> Tuple[Tensor, float, float]:
    """
    Preprocess the image: resize while preserving aspect ratio.
    Returns the preprocessed image and the scaling factors.
    """
    # Get original dimensions
    orig_h, orig_w = orig_image.shape[:2]

    # Define the target size
    target_size = cfg.INPUT.MIN_SIZE_TEST  # e.g., 600
    max_size = cfg.INPUT.MAX_SIZE_TEST  # e.g., 1000

    # Compute scaling factor
    h, w = orig_h, orig_w
    scale = target_size / min(h, w)
    if h * scale > max_size or w * scale > max_size:
        scale = max_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Resize image
    image = cv2.resize(orig_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Convert to torch tensor and normalize
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    image = (image - torch.tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)) / torch.tensor(
        cfg.MODEL.PIXEL_STD
    ).view(-1, 1, 1)

    # Compute scaling factors
    scale_x = new_w / orig_w
    scale_y = new_h / orig_h

    return image, scale_x, scale_y


class BoundingBoxModule:
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
        cfg = get_cfg()

        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        )
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )

        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0  # Detection threshold is 0
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available")

        # Build model
        model = build_model(cfg)
        model.eval()
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        # Get class names
        metadata = MetadataCatalog.get(
            cfg.DATASETS.TRAIN[0] if len(cfg.DATASETS.TRAIN) else "__unused"
        )
        class_names = metadata.get("thing_classes", None)

        self.person_class_id = class_names.index("person") if "person" in class_names else 0
        self.cfg = cfg
        self.model = model

    def run(self, waybetter_images: List[WaybetterImage]) -> None:
        # Process images in batches
        for i in tqdm(
            range(0, len(waybetter_images), self.batch_size), desc="Processing batches"
        ):
            try:
                batch_images = waybetter_images[i : i + self.batch_size]  # noqa
                inputs = []
                original_images: List[Tuple[WaybetterImage, Any]] = []

                for w in batch_images:
                    img_path = w.original_path
                    orig_image = cv2.imread(str(img_path))
                    original_images.append((w, orig_image))

                    # Preprocess the image
                    image = preprocess_image(orig_image, self.cfg)
                    inputs.append(
                        {
                            "image": image,
                            "height": orig_image.shape[0],
                            "width": orig_image.shape[1],
                        }
                    )

                # Move inputs to the appropriate device
                if torch.cuda.is_available():
                    inputs = [
                        {
                            k: v.cuda() if isinstance(v, torch.Tensor) else v
                            for k, v in x.items()
                        }
                        for x in inputs
                    ]

                # Run model inference
                with torch.no_grad():
                    outputs = self.model(inputs)

                # Process outputs
                for (w, _), output in zip(original_images, outputs):
                    # Get predicted classes, scores, and bounding boxes
                    instances = output["instances"].to("cpu")
                    pred_classes = instances.pred_classes.numpy()
                    pred_scores = instances.scores.numpy()
                    pred_boxes = instances.pred_boxes.tensor.numpy()  # Bounding boxes

                    # Collect bounding boxes for detected persons with confidence >= threshold
                    person_boxes = []
                    for cls, score, box in zip(pred_classes, pred_scores, pred_boxes):
                        if cls == self.person_class_id:
                            person_boxes.append((score, box))

                    if person_boxes:
                        # Select the bounding box with highest confidence
                        best_score, best_box = max(person_boxes, key=lambda x: x[0])
                        x1, y1, x2, y2 = best_box
                        w.bounding_box = HumanBoundingBox(
                            x1=int(x1),
                            y1=int(y1),
                            x2=int(x2),
                            y2=int(y2),
                            confidence=float(best_score),
                        )
                    else:
                        # No person detected with confidence >= threshold
                        w.bounding_box = None
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue

    def run_small(self, _images: List[WaybetterImage]) -> None:
        # Process images in batches
        for i in tqdm(range(0, len(_images), self.batch_size), desc="Processing batches"):
            batch_images = _images[i : i + self.batch_size]
            inputs = []
            original_images: List[Tuple[WaybetterImage, Any]] = []
            scaling_factors = []
            time_start = time.time()
            for w in batch_images:
                img_path = w.original_path
                orig_image = cv2.imread(str(img_path))
                original_images.append((w, orig_image))

                # Preprocess the image and get scaling factors
                image, scale_x, scale_y = preprocess_image_small(orig_image, self.cfg)
                scaling_factors.append((scale_x, scale_y))

                inputs.append(
                    {
                        "image": image.to(self.cfg.MODEL.DEVICE),
                        "height": orig_image.shape[0],
                        "width": orig_image.shape[1],
                    }
                )
            time_end = time.time()
            print(f"Preprocessing completed in {time_end - time_start:.4f} seconds.")

            # Run model inference
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model(inputs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Batch forward pass completed in {elapsed_time:.4f} seconds.")

            # Process outputs and scale bounding boxes back
            for (w, orginal), output, (scale_x, scale_y) in zip(
                original_images, outputs, scaling_factors
            ):
                # Get predicted classes, scores, and bounding boxes
                instances = output["instances"].to("cpu")
                pred_classes = instances.pred_classes.numpy()
                pred_scores = instances.scores.numpy()
                pred_boxes = instances.pred_boxes.tensor.numpy()  # Bounding boxes

                # Collect bounding boxes for detected persons with confidence >= threshold
                person_boxes = []
                for cls, score, box in zip(pred_classes, pred_scores, pred_boxes):
                    if cls == self.person_class_id:
                        person_boxes.append((score, box))

                if person_boxes:
                    # Select the bounding box with highest confidence
                    best_score, best_box = max(person_boxes, key=lambda x: x[0])
                    x1, y1, x2, y2 = best_box

                    # Scale bounding boxes back to original image dimensions
                    x1 = x1 / scale_x
                    y1 = y1 / scale_y
                    x2 = x2 / scale_x
                    y2 = y2 / scale_y

                    # Clip coordinates to image dimensions
                    x1 = max(0, min(x1, orginal.shape[1]))
                    y1 = max(0, min(y1, orginal.shape[0]))
                    x2 = max(0, min(x2, orginal.shape[1]))
                    y2 = max(0, min(y2, orginal.shape[0]))

                    w.bounding_box = HumanBoundingBox(
                        x1=int(x1),
                        y1=int(y1),
                        x2=int(x2),
                        y2=int(y2),
                        confidence=float(best_score),
                    )
                else:
                    # No person detected with confidence >= threshold
                    w.bounding_box = None

        print(f"Total images processed: {len(_images)}")