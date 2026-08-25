import os
import sys
import json
import torch
import pandas as pd
import multiprocessing
import argparse
from typing import List, Tuple, Literal, Optional, Dict
from pydantic import BaseModel, validator
from scripts.finetune_densenet import finetune_densenet, evaluate_model

# Constants
VISUAL_BODY_TO_BMI_CSV_PATH = "data/parsed_visual_bmi_dataset.csv"
PRETRAINED_MODEL_PATHS = {
    "densenet_base": "trained_models/no_user_overlap_40_epochs/best_model.ckpt",
    "densenet_large": "trained_models/no_user_overlap_40_epochs_large/best_model.ckpt",
}

# Hyperparameter configurations
LEARNING_RATE_CONFIGS: List[Tuple[float, float, float, float, float, float, float]] = [
    # (head_lr, backbone_lr, max_head_lr, max_backbone_lr, warmup_pct, div_factor, final_div_factor)
    (1e-4, 1e-5, 1e-3, 1e-4, 0.3, 25, 1e4),  # Conservative
    (5e-4, 1e-4, 5e-3, 5e-4, 0.4, 10, 1e3),  # Aggressive
    (2e-4, 2e-5, 2e-3, 2e-4, 0.35, 15, 5e3), # Moderate
    (1e-3, 1e-4, 1e-2, 1e-3, 0.2, 5, 1e2),   # Very aggressive
    (5e-5, 5e-6, 5e-4, 5e-5, 0.5, 50, 1e5),  # Very conservative
]

FREEZE_STRATEGIES: List[Literal["freeze_features", "unfreeze_last_block", "unfreeze_all"]] = [
    "freeze_features",
    "unfreeze_last_block",
    "unfreeze_all",
]

class HyperparameterConfig(BaseModel):
    """Configuration for a single hyperparameter experiment"""
    model_type: Literal["base", "large"]
    freeze_strategy: Literal["freeze_features", "unfreeze_last_block", "unfreeze_all"]
    head_lr: float
    backbone_lr: float
    max_head_lr: float
    max_backbone_lr: float
    warmup_pct: float
    div_factor: float
    final_div_factor: float
    batch_size: int = 32  # Fixed batch size
    epochs: int = 40      # Fixed epochs

    def get_save_dir(self) -> str:
        """Generate a unique save directory name for this config"""
        base_dir = "trained_models/finetuning_hyper_parameter_search"
        model_type = self.model_type
        strategy = self.freeze_strategy
        lr_str = f"h{self.head_lr:.0e}_b{self.backbone_lr:.0e}_mh{self.max_head_lr:.0e}_mb{self.max_backbone_lr:.0e}"
        warmup = f"w{int(self.warmup_pct*100)}"
        div = f"d{int(self.div_factor)}_fd{int(self.final_div_factor)}"
        return os.path.join(base_dir, f"{model_type}_{strategy}_{lr_str}_{warmup}_{div}")

    def save_config(self, save_dir: str) -> None:
        """Save the configuration to a JSON file"""
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.dict(), f, indent=2)

class TrainingJob(BaseModel):
    """Configuration for a single training job"""
    model_type: Literal["base", "large"]
    freeze_strategy: Literal["freeze_features", "unfreeze_last_block", "unfreeze_all"]
    hyperparams: HyperparameterConfig
    assigned_device: Optional[str] = None

    @property
    def base_model_path(self) -> str:
        """Get the path to the base model checkpoint"""
        model_key = f"densenet_{self.model_type}"
        if model_key not in PRETRAINED_MODEL_PATHS:
            raise ValueError(f"No pretrained model path configured for {model_key}")
        path = PRETRAINED_MODEL_PATHS[model_key]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Pretrained model not found at {path}")
        return path

def run_training_job(job: TrainingJob) -> None:
    """Run a single training job with the given configuration"""
    # Setup logging
    save_dir = job.hyperparams.get_save_dir()
    os.makedirs(save_dir, exist_ok=True)
    job.hyperparams.save_config(save_dir)
    
    log_file_path = os.path.join(save_dir, "training.log")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    try:
        # Set CUDA device if specified
        if job.assigned_device:
            os.environ["CUDA_VISIBLE_DEVICES"] = job.assigned_device.split(":")[-1]
        
        # Setup logging
        with open(log_file_path, 'w') as log_file:
            sys.stdout = log_file
            sys.stderr = log_file
            
            print(f"=== Training Job Configuration ===")
            print(f"Model Type: {job.model_type}")
            print(f"Freeze Strategy: {job.freeze_strategy}")
            print(f"Learning Rates: Head={job.hyperparams.head_lr:.2e}, Backbone={job.hyperparams.backbone_lr:.2e}")
            print(f"Max Learning Rates: Head={job.hyperparams.max_head_lr:.2e}, Backbone={job.hyperparams.max_backbone_lr:.2e}")
            print(f"Warmup: {job.hyperparams.warmup_pct*100}%, Div Factor: {job.hyperparams.div_factor}")
            print(f"Final Div Factor: {job.hyperparams.final_div_factor}")
            print(f"Device: {job.assigned_device}")
            print(f"Save Directory: {save_dir}")
            print("================================")

            # Load dataset
            if not os.path.exists(VISUAL_BODY_TO_BMI_CSV_PATH):
                raise FileNotFoundError(f"Dataset CSV not found: {VISUAL_BODY_TO_BMI_CSV_PATH}")
            
            visual_data_df = pd.read_csv(VISUAL_BODY_TO_BMI_CSV_PATH)
            print(f"Loaded dataset with {len(visual_data_df)} rows")

            # Run fine-tuning
            finetune_densenet(
                model_checkpoint_path=job.base_model_path,
                freeze_strategy=job.freeze_strategy,
                visual_body_to_bmi_data=visual_data_df,
                save_dir=save_dir,
                epochs=job.hyperparams.epochs,
                batch_size=job.hyperparams.batch_size,
                absolute_path_col="image_path",
                large_model=(job.model_type == "large"),
            )
            print("Training completed successfully")

    except Exception as e:
        print(f"!!! Exception in training job: {e}")
        import traceback
        print(traceback.format_exc())
        raise
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr

def generate_training_jobs() -> List[TrainingJob]:
    """Generate all training job configurations"""
    jobs: List[TrainingJob] = []
    
    # Get available GPUs
    available_cuda_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    if not available_cuda_devices:
        print("WARNING: No CUDA devices found!")
        available_cuda_devices = ["cuda:0"]  # Fallback to single GPU
    
    # Create jobs for each combination
    job_idx = 0
    for model_type in ["base", "large"]:
        for strategy in FREEZE_STRATEGIES:
            for lr_config in LEARNING_RATE_CONFIGS:
                head_lr, backbone_lr, max_head_lr, max_backbone_lr, warmup_pct, div_factor, final_div_factor = lr_config
                
                hyperparams = HyperparameterConfig(
                    model_type=model_type,  # type: ignore
                    freeze_strategy=strategy,  # type: ignore
                    head_lr=head_lr,
                    backbone_lr=backbone_lr,
                    max_head_lr=max_head_lr,
                    max_backbone_lr=max_backbone_lr,
                    warmup_pct=warmup_pct,
                    div_factor=div_factor,
                    final_div_factor=final_div_factor
                )
                
                job = TrainingJob(
                    model_type=model_type,  # type: ignore
                    freeze_strategy=strategy,  # type: ignore
                    hyperparams=hyperparams,
                    assigned_device=available_cuda_devices[job_idx % len(available_cuda_devices)]
                )
                
                jobs.append(job)
                job_idx += 1
    
    return jobs

def main():
    parser = argparse.ArgumentParser(description="Run hyperparameter search for fine-tuning DenseNet models")
    parser.add_argument("--dry-run", action="store_true", help="Print jobs without executing")
    args = parser.parse_args()

    # Set multiprocessing start method
    try:
        multiprocessing.set_start_method("fork", force=True)
    except RuntimeError:
        pass

    # Generate jobs
    print("\n======= Generating Hyperparameter Search Jobs =======")
    training_jobs = generate_training_jobs()
    
    # Print job summary
    print(f"\nGenerated {len(training_jobs)} training jobs:")
    for i, job in enumerate(training_jobs):
        print(f"\nJob {i+1}:")
        print(f"  Model: {job.model_type}")
        print(f"  Strategy: {job.freeze_strategy}")
        print(f"  Learning Rates: Head={job.hyperparams.head_lr:.2e}, Backbone={job.hyperparams.backbone_lr:.2e}")
        print(f"  Max LRs: Head={job.hyperparams.max_head_lr:.2e}, Backbone={job.hyperparams.max_backbone_lr:.2e}")
        print(f"  Warmup: {job.hyperparams.warmup_pct*100}%, Div Factor: {job.hyperparams.div_factor}")
        print(f"  Device: {job.assigned_device}")
        print(f"  Save Dir: {job.hyperparams.get_save_dir()}")

    if args.dry_run:
        print("\n*** DRY RUN - No jobs will be executed ***")
        return

    # Run jobs in parallel
    num_parallel = min(len(training_jobs), 6)  # Max 6 GPUs
    print(f"\n======= Running {len(training_jobs)} jobs with {num_parallel} parallel processes =======")
    
    with multiprocessing.Pool(processes=num_parallel) as pool:
        results = []
        for job in training_jobs:
            result = pool.apply_async(
                run_training_job,
                (job,),
                error_callback=lambda e: print(f"Error in job: {e}")
            )
            results.append(result)
        
        # Wait for all jobs to complete
        for i, result in enumerate(results):
            try:
                result.get()
                print(f"Job {i+1} completed successfully")
            except Exception as e:
                print(f"Job {i+1} failed: {e}")

    print("\n======= Hyperparameter Search Complete =======")
    print("Check results in: trained_models/finetuning_hyper_parameter_search/")

if __name__ == "__main__":
    main() 