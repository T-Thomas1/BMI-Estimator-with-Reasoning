import os
import sys
import dotenv
sys.path.append(os.getcwd())
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import multiprocessing
import argparse
from multiprocessing import Process
from typing import Dict, List, Optional

from src.scripts.finetune_densenet import evaluate_model
import pandas as pd
import torch

dotenv.load_dotenv()

def find_model_checkpoints(base_dir: str) -> List[Dict]:
    """
    Find all model checkpoints in the given directory structure.
    Returns a list of job configurations for evaluation.
    """
    jobs = []
    model_types = ["base", "large"]
    
    for model_type in model_types:
        model_dir = os.path.join(base_dir, model_type)
        if not os.path.exists(model_dir):
            print(f"Warning: Model directory not found: {model_dir}")
            continue
            
        checkpoint_path = os.path.join(model_dir, "best_model.ckpt")
        if os.path.exists(checkpoint_path):
            jobs.append({
                "checkpoint_path": checkpoint_path,
                "large_model": (model_type == "large"),
                "model_type": model_type
            })
        else:
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
    
    return jobs

def run_evaluation_job(job_dict: Dict, dataset_df: pd.DataFrame):
    """
    Run a single evaluation job on a specific GPU.
    """
    assigned_device = job_dict.get("assigned_device")
    if assigned_device:
        os.environ["CUDA_VISIBLE_DEVICES"] = assigned_device.split(":")[-1]
        print(f"Using GPU: {assigned_device}")
    
    try:
        print(f"\nEvaluating {job_dict['model_type']} model...")
        evaluate_model(
            checkpoint_path=job_dict["checkpoint_path"],
            visual_body_to_bmi_data=dataset_df,
            absolute_path_col="image_path",
            large_model=job_dict["large_model"]
        )
        print(f"Evaluation of {job_dict['model_type']} model completed successfully")
    except Exception as e:
        print(f"Error evaluating {job_dict['model_type']} model: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise

def evaluate_models(
    dataset_path: str,
    models_dir: str = "trained_models/visual_bmi",
    run_parallel: bool = True
):
    """
    Evaluate all trained models in the specified directory.
    
    Args:
        dataset_path: Path to the dataset CSV file
        models_dir: Base directory containing the trained models
        run_parallel: Whether to run evaluations in parallel
    """
    # Load dataset
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    visual_body_to_bmi_data = pd.read_csv(dataset_path)
    print(f"Loaded dataset with {len(visual_body_to_bmi_data)} rows")
    
    # Find model checkpoints
    jobs = find_model_checkpoints(models_dir)
    if not jobs:
        print(f"No model checkpoints found in {models_dir}")
        return
    
    # Check available GPUs
    available_cuda_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    if not available_cuda_devices:
        print("Warning: No CUDA devices found. Evaluation will use CPU.")
        run_parallel = False
    
    # Assign devices to jobs
    for i, job in enumerate(jobs):
        if available_cuda_devices:
            job["assigned_device"] = available_cuda_devices[i % len(available_cuda_devices)]
        else:
            job["assigned_device"] = None
    
    if run_parallel and len(jobs) > 1:
        print(f"Running evaluations in parallel on GPUs: {available_cuda_devices[:len(jobs)]}")
        
        # Create and start processes
        processes = []
        for job in jobs:
            p = Process(
                target=run_evaluation_job,
                args=(job, visual_body_to_bmi_data)
            )
            p.daemon = False
            processes.append(p)
            p.start()
        
        # Wait for all processes to complete
        for i, p in enumerate(processes):
            p.join()
            if p.exitcode != 0:
                print(f"Evaluation of {jobs[i]['model_type']} model failed with exit code {p.exitcode}")
    else:
        print("Running evaluations sequentially")
        for job in jobs:
            run_evaluation_job(job, visual_body_to_bmi_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained VisualBodyToBMI models")
    parser.add_argument("--dataset", type=str, default="data/parsed_visual_bmi_dataset.csv",
                      help="Path to the dataset CSV file")
    parser.add_argument("--models-dir", type=str, default="trained_models/visual_bmi",
                      help="Directory containing the trained models")
    parser.add_argument("--sequential", action="store_true",
                      help="Run evaluations sequentially instead of in parallel")
    args = parser.parse_args()
    
    evaluate_models(
        dataset_path=args.dataset,
        models_dir=args.models_dir,
        run_parallel=not args.sequential
    ) 