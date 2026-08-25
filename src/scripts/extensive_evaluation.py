import os
from pydantic import BaseModel, validator
from typing import Literal, Dict, Optional
import pandas as pd
import torch
from scripts.finetune_densenet import finetune_densenet, evaluate_model
import sys
import traceback
import argparse

# Placeholder for pretrained model paths - REPLACE WITH ACTUAL PATHS
PRETRAINED_DENSENET_BASE_PATH = "trained_models/no_user_overlap_40_epochs/best_model.ckpt"
PRETRAINED_DENSENET_LARGE_PATH = "trained_models/no_user_overlap_40_epochs_large/best_model.ckpt"

# Placeholder for dataset path - REPLACE WITH ACTUAL PATH
VISUAL_BODY_TO_BMI_CSV_PATH = "data/parsed_visual_bmi_dataset.csv" # Or the correct filename

# --- Configuration for model paths ---
# This dictionary maps an identifier to the actual file path.
# The job definition will use the identifier.
pretrained_model_paths: Dict[str, str] = {
    "densenet_base_placeholder": PRETRAINED_DENSENET_BASE_PATH,
    "densenet_large_placeholder": PRETRAINED_DENSENET_LARGE_PATH,
    # Add more pre-trained models here if needed, e.g.:
    # "some_other_model": "path/to/some_other_model.pth"
}

class Job(BaseModel):
    type: Literal["train", "inference"]
    model_type: Literal["base", "large"] # Renamed from model_size
    dataset_identifier: Literal["visual_body_to_bmi_csv"] # Renamed from dataset
    # This field will be used by the job definition to specify WHICH pretrained model to use.
    base_model_identifier: str
    # This field will be populated dynamically before running the job.
    actual_base_model_path: Optional[str] = None
    layer_freeze_strategy: Literal["freeze_features", "unfreeze_last_block", "unfreeze_all"]
    # Field to store assigned CUDA device, e.g., "cuda:0"
    assigned_device: Optional[str] = None


    @validator("actual_base_model_path", always=True)
    def populate_and_check_model_path(cls, v, values):
        if values.get("type") == "train": # Only populate for training jobs that need a base model
            identifier = values.get("base_model_identifier")
            if not identifier:
                raise ValueError("base_model_identifier must be set for a training job.")
            path = pretrained_model_paths.get(identifier)
            if not path:
                raise ValueError(f"No path configured for base_model_identifier: {identifier}")
            if not os.path.exists(path):
                # Only raise critical error if it's a placeholder path AFTER user should have replaced it
                if "path/to/your" in path: # Check if it's still a placeholder
                    print(f"Warning: Placeholder path used for {identifier}: {path}. Please update.")
                    # Allow to proceed for setup, but it will fail in finetune_densenet if not updated
                else:
                    raise FileNotFoundError(f"Model file not found for identifier {identifier}: {path}")
            return path
        return v # For inference or if not applicable

# Removed the old TrainedModel class as its functionality is merged or handled differently.

def run_job_on_device(job_dict: dict): # Expecting a dict that can be parsed into Job or has evaluate keys
    # This function will run in a separate process.
    # We need to re-import and re-setup things that are not picklable
    # or are specific to the child process, like CUDA device visibility.

    # --- Original stdout and stderr for restoration ---
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file_path = None
    log_file = None
    job_type = job_dict.get("type")
    assigned_device = job_dict.get("assigned_device")

    try:
        # Set CUDA device for this specific job process
        if assigned_device:
            os.environ["CUDA_VISIBLE_DEVICES"] = assigned_device.split(":")[-1]
        
        if job_type == "train":
            job = Job(**job_dict) # Parse training job details

            # --- Setup Training save directory and log redirection --- 
            save_dir_base = "trained_models/finetuned/visual_body_to_bmi"
            model_type_str = str(job.model_type).replace(" ", "_")
            strategy_str = str(job.layer_freeze_strategy).replace(" ", "_")
            job_save_dir = os.path.join(save_dir_base, f"{model_type_str}_{strategy_str}")
            os.makedirs(job_save_dir, exist_ok=True)
            
            log_file_path = os.path.join(job_save_dir, "training.log")
            log_file = open(log_file_path, 'w')
            sys.stdout = log_file
            sys.stderr = log_file
            print(f"--- Training Job Log for {job.model_type} - {job.layer_freeze_strategy} ---")
            print(f"Assigned device: {job.assigned_device}, CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
            print(f"Results will be saved in: {job_save_dir}")
            # --- End Training log redirection setup ---

            print(f"Starting training job: {job.model_type} model, strategy: {job.layer_freeze_strategy} on {job.assigned_device or 'default device'}")

            # --- Training Specific Logic --- 
            if not job.actual_base_model_path:
                print(f"Error: actual_base_model_path not populated for job: {job.model_type}, {job.layer_freeze_strategy}")
                raise ValueError("actual_base_model_path is missing.")

            if "path/to/your" in job.actual_base_model_path:
                print(f"Critical: Pretrained model path for {job.base_model_identifier} is still a placeholder: {job.actual_base_model_path}. Please update.")
                raise FileNotFoundError(
                    f"Critical: Pretrained model path for {job.base_model_identifier} is still a placeholder: "
                    f"{job.actual_base_model_path}. Please update it in extensive_evaluation.py."
                )
            if not os.path.exists(job.actual_base_model_path):
                print(f"Error: Model file not found: {job.actual_base_model_path}")
                raise FileNotFoundError(f"Model file not found: {job.actual_base_model_path}")

            if job.dataset_identifier == "visual_body_to_bmi_csv":
                if not os.path.exists(VISUAL_BODY_TO_BMI_CSV_PATH):
                    print(f"Error: Dataset CSV not found: {VISUAL_BODY_TO_BMI_CSV_PATH}")
                    raise FileNotFoundError(f"Dataset CSV not found: {VISUAL_BODY_TO_BMI_CSV_PATH}")
                # Removed redundant warning check for default path

                visual_data_df = pd.read_csv(VISUAL_BODY_TO_BMI_CSV_PATH)
                print(f"Loaded dataset {VISUAL_BODY_TO_BMI_CSV_PATH} with {len(visual_data_df)} rows for training.")

                finetune_densenet(
                    model_checkpoint_path=job.actual_base_model_path,
                    freeze_strategy=job.layer_freeze_strategy,
                    visual_body_to_bmi_data=visual_data_df,
                    save_dir=job_save_dir, # Pass the specific job's save directory
                    epochs=40, 
                    batch_size=32, 
                    absolute_path_col="image_path", # Make sure this matches your CSV header
                    large_model=(job.model_type == "large"),
                )
                print(f"Finished training job: {job.model_type} model, strategy: {job.layer_freeze_strategy}")
            else:
                print(f"Dataset {job.dataset_identifier} not implemented for training.")
            # --- End Training Specific Logic ---

        elif job_type == "evaluate":
            # --- Setup Evaluation save directory and log redirection ---
            fine_tuned_model_path = job_dict.get("fine_tuned_model_path")
            if not fine_tuned_model_path or not os.path.exists(fine_tuned_model_path):
                 # Log to original stdout/stderr if path is bad, as we can't create log file
                 original_stderr.write(f"Error: Fine-tuned model path not found or invalid for evaluation: {fine_tuned_model_path}\n")
                 raise FileNotFoundError(f"Fine-tuned model not found for evaluation: {fine_tuned_model_path}")

            job_save_dir = os.path.dirname(fine_tuned_model_path)
            log_file_path = os.path.join(job_save_dir, "evaluation.log")
            log_file = open(log_file_path, 'w')
            sys.stdout = log_file
            sys.stderr = log_file
            model_type_str = job_dict.get("model_type", "unknown")
            print(f"--- Evaluation Job Log for {model_type_str} model ({os.path.basename(job_save_dir)}) ---")
            print(f"Assigned device: {assigned_device}, CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
            print(f"Evaluating model: {fine_tuned_model_path}")
            print(f"Results will be saved in: {job_save_dir}")
            # --- End Evaluation log redirection setup ---
            
            print(f"Starting evaluation job for model: {fine_tuned_model_path} on {assigned_device or 'default device'}")

            # --- Evaluation Specific Logic ---
            dataset_identifier = job_dict.get("dataset_identifier")
            absolute_path_col = job_dict.get("absolute_path_col")
            large_model = (job_dict.get("model_type") == "large")
            
            if dataset_identifier == "visual_body_to_bmi_csv":
                if not os.path.exists(VISUAL_BODY_TO_BMI_CSV_PATH):
                    print(f"Error: Dataset CSV not found: {VISUAL_BODY_TO_BMI_CSV_PATH}")
                    raise FileNotFoundError(f"Dataset CSV not found: {VISUAL_BODY_TO_BMI_CSV_PATH}")
                
                visual_data_df = pd.read_csv(VISUAL_BODY_TO_BMI_CSV_PATH)
                print(f"Loaded dataset {VISUAL_BODY_TO_BMI_CSV_PATH} with {len(visual_data_df)} rows for evaluation.")

                evaluate_model(
                    checkpoint_path=fine_tuned_model_path,
                    visual_body_to_bmi_data=visual_data_df,
                    absolute_path_col=absolute_path_col,
                    large_model=large_model
                    # Batch size is hardcoded to 1 inside evaluate_model
                )
                print(f"Finished evaluation job for model: {fine_tuned_model_path}")
            else:
                print(f"Dataset {dataset_identifier} not implemented for evaluation.")
            # --- End Evaluation Specific Logic ---

        elif job_type == "inference":
            # Original placeholder - can be removed or implemented later
            original_stdout.write("Inference job type not fully implemented yet.\n")
            pass
        
        else:
            original_stderr.write(f"Error: Unknown job type '{job_type}' in job_dict: {job_dict}\n")
            raise ValueError(f"Unknown job type: {job_type}")

    except Exception as e:
        # Log exception (will go to log file if redirection happened, otherwise original stderr)
        print(f"!!! Exception in job {job_dict}: {e}") 
        print(traceback.format_exc()) 
        raise # Re-raise the exception
    finally:
        # --- Restore stdout and stderr ---
        if log_file and not log_file.closed:
            log_file.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        # --- End restoration ---

# Main execution block
if __name__ == "__main__":
    import multiprocessing
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run evaluation jobs for fine-tuned DenseNet models.")
    parser.add_argument("--dry-run", action="store_true", 
                        help="Print the evaluation jobs that would be run, but do not execute them.")
    args = parser.parse_args()
    # --- End Argument Parsing ---

    # Set multiprocessing start method
    try:
        multiprocessing.set_start_method("fork", force=True)
        print("Set multiprocessing start method to 'fork'.")
    except RuntimeError:
        print("Multiprocessing start method already set or 'fork' not available/safe.")
        pass 

    # ======= PHASE 1: Define and Run Training Jobs (SKIPPED) =======
    print("\n======= PHASE 1: CONFIGURING TRAINING JOBS =======")
    model_types = ["base", "large"]
    freeze_strategies = ["freeze_features", "unfreeze_last_block", "unfreeze_all"]
    
    training_jobs_to_run_configs = []
    available_cuda_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] 
    if not available_cuda_devices:
        print("CRITICAL: No CUDA devices found by PyTorch. Training jobs will lack specific GPU assignment.")
    
    job_idx = 0
    for model_t in model_types:
        for strategy in freeze_strategies:
            base_model_id = "densenet_base_placeholder" if model_t == "base" else "densenet_large_placeholder"
            
            if base_model_id not in pretrained_model_paths:
                print(f"ERROR: base_model_identifier '{base_model_id}' not found in pretrained_model_paths. Skipping job.")
                continue
    
            assigned_device_val: Optional[str] = None
            if available_cuda_devices:
                assigned_device_val = available_cuda_devices[job_idx % len(available_cuda_devices)]
            
            job_config = {
                "type": "train",
                "model_type": model_t,
                "dataset_identifier": "visual_body_to_bmi_csv",
                "base_model_identifier": base_model_id,
                "layer_freeze_strategy": strategy,
                "assigned_device": assigned_device_val 
            }
            training_jobs_to_run_configs.append(job_config)
            job_idx += 1
    
    # Validate training job configurations
    valid_training_job_configs = []
    print("\nValidating training job configurations...")
    for job_data in training_jobs_to_run_configs:
        try:
            # Explicitly create Job instance for validation
            job_instance = Job(
                type=job_data['type'],
                model_type=job_data['model_type'],
                dataset_identifier=job_data['dataset_identifier'],
                base_model_identifier=job_data['base_model_identifier'],
                layer_freeze_strategy=job_data['layer_freeze_strategy'],
                assigned_device=job_data.get('assigned_device') 
            )
            valid_training_job_configs.append(job_data)
        except (ValueError, FileNotFoundError, TypeError) as e: 
            print(f"Configuration error for training job {job_data.get('model_type')}, {job_data.get('layer_freeze_strategy')}: {e}")
            print("This job will be skipped. Please check paths and configurations.")
    
    if not valid_training_job_configs:
        print("No valid training jobs to run. Exiting.")
        sys.exit(1) # Exit if no training jobs can run
    else:
        print(f"\nPrepared {len(valid_training_job_configs)} training jobs for execution:")
        for i, job_data in enumerate(valid_training_job_configs):
             # Explicitly create Job instance for display
            job_instance_for_display = Job(
                type=job_data['type'],
                model_type=job_data['model_type'],
                dataset_identifier=job_data['dataset_identifier'],
                base_model_identifier=job_data['base_model_identifier'],
                layer_freeze_strategy=job_data['layer_freeze_strategy'],
                assigned_device=job_data.get('assigned_device') 
            )
            print(f"  Training Job {i+1}: Model={job_instance_for_display.model_type}, Strategy={job_instance_for_display.layer_freeze_strategy}, BaseModelID={job_instance_for_display.base_model_identifier}, ActualPath={job_instance_for_display.actual_base_model_path}, Device={job_instance_for_display.assigned_device}")
        print("\n")
    
        # Run training jobs using a Pool
        num_parallel_processes = min(len(available_cuda_devices) if available_cuda_devices else 1, len(valid_training_job_configs), 4)
        print(f"======= PHASE 1: RUNNING TRAINING JOBS ({num_parallel_processes} parallel processes) =======")
        
        with multiprocessing.Pool(processes=num_parallel_processes) as pool:
            training_results = []
            for job_data in valid_training_job_configs:
                result = pool.apply_async(run_job_on_device, (job_data,), error_callback=lambda e: print(f"Error in training worker process for job {job_data}: {e}"))
                training_results.append(result)
            
            # Wait for all training jobs to complete
            successful_train_configs = [] # This needs to be defined even if training is skipped
            for i, result in enumerate(training_results):
                job_data = valid_training_job_configs[i]
                try:
                    result.get() 
                    print(f"Training Job {i+1} (config: {job_data}) completed successfully.")
                    successful_train_configs.append(job_data) # Keep track of successful configs
                except Exception as e:
                    print(f"Training Job {i+1} (config: {job_data}) failed with error: {e}")
    
    print("\n======= PHASE 1: TRAINING COMPLETE =======")
    # ======= END OF PHASE 1 SKIPPED BLOCK =======

    # ======= PHASE 2: Define and Run Evaluation Jobs =======
    print("\n======= PHASE 2: CONFIGURING EVALUATION JOBS =======")
    evaluation_jobs_to_run_configs = []
    
    # Define potential model configurations directly
    model_types = ["base", "large"]
    freeze_strategies = ["freeze_features", "unfreeze_last_block", "unfreeze_all"]
    available_cuda_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] 
    if not available_cuda_devices:
        print("Warning: No CUDA devices found by PyTorch. Evaluation jobs will lack specific GPU assignment.")

    print(f"Scanning for existing fine-tuned models to evaluate...")
    job_idx = 0 # Index for device assignment
    for model_t in model_types:
        for strategy in freeze_strategies:
            # Determine the expected output path based on convention
            save_dir_base = "trained_models/finetuned/visual_body_to_bmi"
            model_type_str = str(model_t).replace(" ", "_")
            strategy_str = str(strategy).replace(" ", "_")
            job_save_dir = os.path.join(save_dir_base, f"{model_type_str}_{strategy_str}")
            fine_tuned_model_path = os.path.join(job_save_dir, "best_model.ckpt")

            if not os.path.exists(fine_tuned_model_path):
                # print(f"Info: Fine-tuned model {fine_tuned_model_path} not found. Skipping evaluation.") # Optional: Reduce noise
                continue
            else:
                 print(f"Found existing model: {fine_tuned_model_path}")

            # Assign device
            eval_assigned_device_val: Optional[str] = None # Renamed variable
            if available_cuda_devices:
                eval_assigned_device_val = available_cuda_devices[job_idx % len(available_cuda_devices)]

            # Create evaluation job config
            eval_job_config = {
                "type": "evaluate",
                "fine_tuned_model_path": fine_tuned_model_path,
                "model_type": model_t, 
                "dataset_identifier": "visual_body_to_bmi_csv", # Assuming evaluation uses the same dataset
                "absolute_path_col": "image_path", # Assuming same column name
                "assigned_device": eval_assigned_device_val
            }
            evaluation_jobs_to_run_configs.append(eval_job_config)
            job_idx += 1

    if not evaluation_jobs_to_run_configs:
        print("No existing fine-tuned models found in expected locations to evaluate. Exiting evaluation phase.")
    else:
        print(f"\nPrepared {len(evaluation_jobs_to_run_configs)} evaluation jobs for execution:")
        for i, job_data in enumerate(evaluation_jobs_to_run_configs):
             # Simple print, removed Pydantic validation/display which caused issues
             print(f"  Evaluation Job {i+1}: ModelPath={job_data['fine_tuned_model_path']}, Device={job_data['assigned_device']}")
        print("\n")

        # --- Dry Run Check ---
        if args.dry_run:
            print("*** DRY RUN ENABLED ***")
            print("Jobs listed above would be executed. Exiting now without running jobs.")
            sys.exit(0)
        # --- End Dry Run Check ---
        
        # Run evaluation jobs using a Pool
        num_parallel_processes = min(len(available_cuda_devices) if available_cuda_devices else 1, len(evaluation_jobs_to_run_configs), 4) # Reuse logic for pool size
        print(f"======= PHASE 2: RUNNING EVALUATION JOBS ({num_parallel_processes} parallel processes) =======")

        with multiprocessing.Pool(processes=num_parallel_processes) as pool:
            evaluation_results = []
            for job_data in evaluation_jobs_to_run_configs:
                result = pool.apply_async(run_job_on_device, (job_data,), error_callback=lambda e: print(f"Error in evaluation worker process for job {job_data}: {e}"))
                evaluation_results.append(result)

            # Wait for all evaluation jobs to complete
            for i, result in enumerate(evaluation_results):
                job_data = evaluation_jobs_to_run_configs[i]
                try:
                    result.get()
                    print(f"Evaluation Job {i+1} (model: {job_data['fine_tuned_model_path']}) completed successfully.")
                except Exception as e:
                    print(f"Evaluation Job {i+1} (model: {job_data['fine_tuned_model_path']}) failed with error: {e}")

    print("\n======= PHASE 2: EVALUATION COMPLETE =======")
    print("\nEvaluation script finished.")
    print(f"Check evaluation output directories starting from: {os.path.join(os.getcwd(), 'trained_models/finetuned/visual_body_to_bmi')}")