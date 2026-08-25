import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src"))

from src.densenet.Densenet import SEDensenet121  # type: ignore # noqa: E402
from src.models.densenet import densenet
from densenet.utils.train import Trainer  # type: ignore # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.optim as optim  # noqa: E402
import torch.utils  # noqa: E402
import torch.utils.data  # noqa: E402
import torch.utils.data.dataloader  # type: ignore # noqa: E402
from models.densenet.densenet_dataloader import WaybetterDataset
from src.helpers.load_waybetter_db import load_waybetter_db
from src.helpers.split_dataset import split_dataframe, split_dataframe_without_user_overlap
import dotenv
import pandas as pd  # noqa: E402


dotenv.load_dotenv()

PHOTOS_PATH = os.environ.get("PHOTOS_DIR")

CHECKPOINT_FOLDER = "trained_models/face_only_large"
LARGE_MODEL = True
DATABASE_PATH = "data/filtered_datasets/filtered_data_december.db"


def forward_pass(
    checkpoint_path: str,
    dataloader: torch.utils.data.DataLoader,
    large_model: bool,
    device: str = "cuda",
) -> Trainer:
    # Load the model architecture
    if large_model:
        # Load the large model
        model = densenet.model_large
    else:
        model = SEDensenet121()

    # Move the model to the desired device
    DEVICE = torch.device(device)
    model.to(DEVICE)

    # Load trained weights from the checkpoint if provided
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    criterion = nn.MSELoss().to(DEVICE)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1,  # Meaningless learning rate, since we are only doing inference
        weight_decay=1,  # Meaningless weight decay value, since we are only doing inference
    )
    trainer = Trainer(model, DEVICE, optimizer, criterion, save_dir="_")
    trainer.load(checkpoint_path)

    trainer.test(dataloader, sex="diff")

    return trainer


### Torso Data
# torso_df = pd.read_csv('/home/rajiv/rajiv-old/DigitalScale/data/filtered_datasets/filtered_data_december_with_torso_image.csv')
# torso_df = torso_df.dropna(subset=['cropped_image_path'])
# # # Drop original image path column to be sure we're using the cropped image path
# torso_df = torso_df.drop(columns=['photo'])

# # # Split the dataset into train, validation and test sets
# torso_df = split_dataframe_without_user_overlap(torso_df)

# test_data = torso_df[torso_df['partition'] == 'test']
# test_dataset = WaybetterDataset(test_data, absolute_path_col="cropped_image_path")

### Face Data
# face_df = pd.read_csv(
#     "/home/rajiv/rajiv-old/DigitalScale/data/filtered_datasets/filtered_data_december_with_face_only.csv"
# )
# face_df = face_df.dropna(subset=["face_only_photo_path"])
# # Drop original image path column to be sure we're using the cropped image path
# face_df = face_df.drop(columns=["photo"])
# # Split the dataset into train, validation and test sets
# face_df = split_dataframe_without_user_overlap(face_df)
# test_data = face_df[face_df["partition"] == "test"]
# test_dataset = WaybetterDataset(test_data, absolute_path_col="face_only_photo_path")


### Regular DATA
# waybetter_df = split_dataframe_without_user_overlap(load_waybetter_db(DATABASE_PATH))
# test_data = waybetter_df[waybetter_df['partition'] == 'test']
# test_dataset = WaybetterDataset(test_data, PHOTOS_PATH)


## Run the forward pass
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, shuffle=True, num_workers=1
)


trainer = forward_pass(CHECKPOINT_FOLDER + "/best_model.ckpt", test_loader, LARGE_MODEL)
# Write the results to a CSV file
results = trainer.output_df
results.to_csv(CHECKPOINT_FOLDER + "/results.csv", index=False)