import os
import sys
import dotenv
sys.path.append(os.getcwd())
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse

from src.models.densenet.densenet_dataloader import get_dataloader
from src.models.densenet.densenet_trainer import Trainer
from src.models.densenet import densenet
from src.helpers.split_dataset import split_visual_bmi_dataframe
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict

dotenv.load_dotenv()

def run_training_job(job_dict: Dict):
    """
    Run a single training job on a specific GPU.
    This function will be called in a separate process.
    """
    # Set CUDA device for this specific job process
    assigned_device = job_dict.get("assigned_device")
    if assigned_device:
        os.environ["CUDA_VISIBLE_DEVICES"] = assigned_device.split(":")[-1]
        print(f"Using GPU: {assigned_device}")
    
    # Get job parameters
    dataset_path = job_dict["dataset_path"]
    save_dir = job_dict["save_dir"]
    epochs = job_dict["epochs"]
    batch_size = job_dict["batch_size"]
    absolute_path_col = job_dict["absolute_path_col"]
    large_model = job_dict["large_model"]
    
    # Create save directory and setup logging
    os.makedirs(save_dir, exist_ok=True)
    log_file_path = os.path.join(save_dir, "training.log")
    
    # Redirect stdout/stderr to log file
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file = open(log_file_path, 'w')
    sys.stdout = log_file
    sys.stderr = log_file
    
    try:
        print(f"=== Starting training job on {assigned_device} ===")
        print(f"Model type: {'DenseNet201' if large_model else 'DenseNet121'}")
        print(f"Save directory: {save_dir}")
        
        # Load and prepare data
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        visual_body_to_bmi_data = pd.read_csv(dataset_path)
        print(f"Loaded dataset with {len(visual_body_to_bmi_data)} rows")
        
        # Split dataset
        visual_body_to_bmi_df = split_visual_bmi_dataframe(visual_body_to_bmi_data)
        train_loader, val_loader, test_loader = get_dataloader(
            visual_body_to_bmi_df, 
            batch_size=batch_size, 
            num_workers=4,
            absolute_path_col=absolute_path_col
        )
        
        # Initialize model
        if large_model:
            model = densenet.model_large
            densenet.load_pretrained_densenet201(model)
            print("Using DenseNet201 (large model) architecture")
        else:
            model = densenet.model
            densenet.load_pretrained_densenet(model)
            print("Using DenseNet121 (base model) architecture")
        
        # Setup training
        DEVICE = torch.device("cuda")
        model.to(DEVICE)
        
        # Training hyperparameters
        LR = 0.0001
        WEIGHT_DECAY = 0.0001
        
        criterion = nn.MSELoss().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        
        # Initialize trainer and start training
        trainer = Trainer(model, DEVICE, optimizer, criterion, save_dir=save_dir)
        print(f"Starting training for {epochs} epochs with batch size {batch_size}")
        
        trainer.Loop(epochs, train_loader, val_loader, scheduler)
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error in training job: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise
    finally:
        # Restore stdout/stderr and close log file
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()

def train_visual_bmi_densenet(
    dataset_path: str,
    save_dir_base: str = "trained_models/visual_bmi",
    epochs: int = 40,
    batch_size: int = 32,
    absolute_path_col: str = "image_path",
    run_parallel: bool = True
):
    """
    Train both base and large models, optionally in parallel on different GPUs.
    
    Args:
        dataset_path: Path to the dataset CSV file
        save_dir_base: Base directory for saving models
        epochs: Number of training epochs
        batch_size: Batch size for training
        absolute_path_col: Column name for absolute image paths
        run_parallel: Whether to run models in parallel on different GPUs
    """
    # Check available GPUs
    available_cuda_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    if not available_cuda_devices:
        print("Warning: No CUDA devices found. Training will use CPU.")
        run_parallel = False
    
    # Prepare job configurations
    jobs = [
        {
            "dataset_path": dataset_path,
            "save_dir": os.path.join(save_dir_base, "base"),
            "epochs": epochs,
            "batch_size": batch_size,
            "absolute_path_col": absolute_path_col,
            "large_model": False,
            "assigned_device": available_cuda_devices[0] if available_cuda_devices else None
        },
        {
            "dataset_path": dataset_path,
            "save_dir": os.path.join(save_dir_base, "large"),
            "epochs": epochs,
            "batch_size": batch_size,
            "absolute_path_col": absolute_path_col,
            "large_model": True,
            "assigned_device": available_cuda_devices[1] if len(available_cuda_devices) > 1 else available_cuda_devices[0] if available_cuda_devices else None
        }
    ]
    
    if run_parallel and len(available_cuda_devices) >= 2:
        print(f"Running training jobs in parallel on GPUs: {available_cuda_devices[:2]}")
        
        # Create and start processes
        processes = []
        for job in jobs:
            # Create a non-daemon process
            p = Process(target=run_training_job, args=(job,))
            p.daemon = False  # Explicitly set to non-daemon
            processes.append(p)
            p.start()
        
        # Wait for all processes to complete
        for i, p in enumerate(processes):
            p.join()
            if p.exitcode == 0:
                print(f"Training job {i+1} completed successfully")
            else:
                print(f"Training job {i+1} failed with exit code {p.exitcode}")
    else:
        print("Running training jobs sequentially")
        for i, job in enumerate(jobs):
            print(f"\nStarting training job {i+1}...")
            run_training_job(job)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DenseNet models on VisualBodyToBMI dataset")
    parser.add_argument("--dataset", type=str, default="data/parsed_visual_bmi_dataset.csv",
                      help="Path to the dataset CSV file")
    parser.add_argument("--save-dir", type=str, default="trained_models/visual_bmi",
                      help="Base directory for saving models")
    parser.add_argument("--epochs", type=int, default=40,
                      help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                      help="Batch size for training")
    parser.add_argument("--sequential", action="store_true",
                      help="Run training jobs sequentially instead of in parallel")
    args = parser.parse_args()
    
    train_visual_bmi_densenet(
        dataset_path=args.dataset,
        save_dir_base=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        run_parallel=not args.sequential
    ) 