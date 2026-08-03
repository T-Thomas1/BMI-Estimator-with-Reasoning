import os
import torch
from dotenv import load_dotenv
from dataclasses import dataclass
from pathlib import Path

load_dotenv()

@dataclass
class Config:
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    MODELS_DIR: Path = BASE_DIR / "models" / "pretrained"
    DATA_DIR: Path = BASE_DIR / "data"
    OUTPUT_DIR: Path = BASE_DIR / "outputs"

    # Model settings
    LLAVA_MODEL: str = "llava-hf/llava-1.5-7b-hf"
    SAM_MODEL_TYPE: str = "vit_h"
    SAM_CHECKPOINT: str = "sam_vit_h_4b8939.pth"

    # Training settings
    RANDOM_SEED: int = 42
    TEST_SIZE: float = 0.2

    # API keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Device
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

config = Config()