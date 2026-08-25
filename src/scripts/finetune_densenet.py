import os
import sys
import dotenv
sys.path.append(os.getcwd())
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from src.models.densenet.densenet_dataloader import get_dataloader
from src.models.densenet.densenet_trainer import Trainer
from src.models.densenet.densenet_trainer import Trainer as EvaluationTrainer
from src.models.densenet import densenet
from src.helpers.split_dataset import split_visual_bmi_dataframe
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Literal
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

dotenv.load_dotenv()


def evaluate_model(checkpoint_path: str,
                  visual_body_to_bmi_data: pd.DataFrame,
                  absolute_path_col: Optional[str] = None,
                  large_model: bool = False) -> None:
    """
    Perform forward pass on test set using EvaluationTrainer and save results.
    Follows the approach of densenet_forwardpass.py for evaluation.
    Results are saved in the same directory as the checkpoint_path.
    Uses a fixed batch size of 1 for evaluation.
    
    Args:
        checkpoint_path: Path to the model checkpoint (from finetune_densenet training)
        visual_body_to_bmi_data: DataFrame containing the dataset for test loader
        absolute_path_col: Column name for absolute image paths if needed
        large_model: Whether to use the large model architecture
    """
    DEVICE = torch.device("cuda")
    save_dir = os.path.dirname(checkpoint_path) # Derive save_dir
    EVAL_BATCH_SIZE = 1 # Hardcoded batch size for evaluation
    
    # 1. Load and split the dataset to get test_loader
    visual_body_to_bmi_df = split_visual_bmi_dataframe(visual_body_to_bmi_data)
    _, _, test_loader = get_dataloader(
        visual_body_to_bmi_df,
        batch_size=EVAL_BATCH_SIZE, # Use hardcoded value
        num_workers=0,
        absolute_path_col=absolute_path_col
    )
    
    # 2. Load model architecture
    if large_model:
        model_arch = densenet.model_large
    else:
        model_arch = densenet.model
    
    model_arch.to(DEVICE)
    
    # 3. Load state_dict from our checkpoint into the model architecture
    if os.path.isfile(checkpoint_path):
        print(f"Loading model state_dict from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        if "state_dict" in checkpoint:
            model_arch.load_state_dict(checkpoint["state_dict"])
        else: # Fallback if the checkpoint is just the state_dict
            model_arch.load_state_dict(checkpoint)
        model_arch.eval() # Set to evaluation mode
    else:
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # 4. Instantiate EvaluationTrainer (from densenet.utils.train)
    # Dummy criterion and optimizer for inference, as in densenet_forwardpass.py
    criterion = nn.MSELoss().to(DEVICE)
    optimizer = optim.Adam(
        model_arch.parameters(), # Pass all params, requires_grad status doesn't matter for dummy optimizer
        lr=1, # Meaningless
        weight_decay=1 # Meaningless
    )
    
    # Use a minimal save_dir for the EvaluationTrainer itself, as it might try to write logs/checkpoints
    # The actual results will be saved to the main `save_dir` derived above.
    # `densenet_forwardpass.py` uses `save_dir="_"` for its trainer.
    evaluation_trainer_internal_save_dir = os.path.join(save_dir, "eval_trainer_temp")
    os.makedirs(evaluation_trainer_internal_save_dir, exist_ok=True)

    evaluation_trainer = EvaluationTrainer(
        model_arch, DEVICE, optimizer, criterion, save_dir=evaluation_trainer_internal_save_dir
    )
    
    # Call `load` on the EvaluationTrainer, as done in densenet_forwardpass.py
    # This might load other metadata or ensure the model is correctly set up within the trainer
    print(f"Calling EvaluationTrainer.load() with checkpoint: {checkpoint_path}")
    evaluation_trainer.load(checkpoint_path)
    
    # 5. Run evaluation on test set using EvaluationTrainer
    print("\nRunning evaluation on test set using EvaluationTrainer...")
    evaluation_trainer.test(test_loader, sex="diff") # Pass 'sex="diff"' as in densenet_forwardpass.py
    
    # 6. Get results from EvaluationTrainer.output_df and save
    if hasattr(evaluation_trainer, 'output_df') and evaluation_trainer.output_df is not None:
        results_df = evaluation_trainer.output_df
        
        # Ensure columns exist and handle potential different naming if needed
        if 'target' in results_df.columns and 'output' in results_df.columns:
            actual = results_df['target']
            predicted = results_df['output']
            
            # Calculate MAE using sklearn
            mae = mean_absolute_error(actual, predicted)
            
            # Calculate MAPE using sklearn 
            # sklearn's function handles division by zero implicitly by ignoring samples where actual is zero.
            # We might still want to check for cases where *all* actuals are zero, though unlikely for BMI.
            try:
                mape = mean_absolute_percentage_error(actual, predicted) * 100
                 # Check if the result is excessively large (can happen if actual values are tiny)
                if mape > 1e6: # Arbitrary large threshold
                    print(f"Warning: Calculated MAPE is very large ({mape:.2f}%). Check actual BMI values near zero.")
                    # Optionally set to NaN or keep the large value depending on desired behavior
                    # mape = np.nan 
            except ValueError: # Should not happen if inputs are numeric arrays, but as safeguard
                 print("Warning: Could not calculate MAPE due to invalid input values.")
                 mape = np.nan

            # If actual contains only zeros (or values close to zero), MAPE would be inf or NaN.
            # Check for NaN result specifically if needed.
            if np.isnan(mape):
                 print("Warning: MAPE calculation resulted in NaN. Check actual BMI values.")

            print(f"\nEvaluation Metrics:")
            print(f"  MAE:  {mae:.4f}")
            print(f"  MAPE: {mape:.2f}%" if not np.isnan(mape) else "  MAPE: NaN")

            # Save metrics to a summary file
            summary_path = os.path.join(save_dir, "evaluation_summary.txt")
            try:
                with open(summary_path, 'w') as f:
                    f.write(f"Evaluation Summary for {os.path.basename(checkpoint_path)}\n")
                    f.write("========================================\n")
                    f.write(f"Mean Absolute Error (MAE):  {mae:.4f}\n")
                    f.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%\n" if not np.isnan(mape) else "Mean Absolute Percentage Error (MAPE): NaN\n")
                print(f"Evaluation summary saved to: {summary_path}")
            except IOError as e:
                print(f"Error writing evaluation summary to {summary_path}: {e}")

        else:
            print("Warning: Could not find 'target' or 'output' columns in output_df. Cannot calculate MAE/MAPE.")

        # Save detailed results CSV
        results_path = os.path.join(save_dir, "test_results.csv")
        results_df.to_csv(results_path, index=False)
        print(f"Detailed test results saved to: {results_path}")
    else:
        print("Warning: EvaluationTrainer did not produce 'output_df'. Test results not saved.")

def finetune_densenet(model_checkpoint_path: str,
                      freeze_strategy: Literal["freeze_features", "unfreeze_last_block", "unfreeze_all"],
                      visual_body_to_bmi_data: pd.DataFrame,
                      save_dir: str = "save_dir",
                      epochs: int = 20,
                      batch_size: int = 32,
                      absolute_path_col: Optional[str] = None,
                      large_model: bool = False,
                      ): # Path to load pre-trained model weights

    visual_body_to_bmi_df = split_visual_bmi_dataframe(visual_body_to_bmi_data)
    train_loader, val_loader, test_loader = get_dataloader(visual_body_to_bmi_df, batch_size=batch_size, num_workers=0, absolute_path_col=absolute_path_col)

    if large_model:
        model = densenet.model_large
    else:
        model = densenet.model

    DEVICE = torch.device("cuda")
    model.to(DEVICE)



    if os.path.isfile(model_checkpoint_path):
        print(f"Loading checkpoint from {model_checkpoint_path}")
        checkpoint = torch.load(model_checkpoint_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        raise FileNotFoundError(f"Checkpoint file not found: {model_checkpoint_path}")

    # Freeze layers based on strategy
    if freeze_strategy == "freeze_features":
        for param in model.features.parameters():
            param.requires_grad = False
        # Ensure classifier is trainable
        model.features.eval()
        for m in model.features.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        for param in model.classifier.parameters():
            param.requires_grad = True
        print("Freezing feature extractor, training only classifier.")
    elif freeze_strategy == "unfreeze_last_block":
        # Freeze all features first
        for param in model.features.parameters():
            param.requires_grad = False
        model.features.eval()
        for m in model.features.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        unfreeze_layers_names = []
        # Unfreeze the last dense block and the transition layer before it (if applicable) + final norm
        # For torchvision densenet, layers are named like 'denseblock4', 'transition3', 'norm5'
        
        # Identify layers to unfreeze by checking attributes
        # DenseNet features are typically: conv0, norm0, relu0, pool0, denseblock1, transition1, ..., denseblockN, transitionN-1, norm_final
        
        # Trying to find the last denseblock (e.g., denseblock4)
        last_denseblock = None
        last_denseblock_name = "" # Initialize here
        if hasattr(model.features, 'denseblock4'):
            last_denseblock = model.features.denseblock4
            unfreeze_layers_names.append('denseblock4')
            last_denseblock_name = 'denseblock4' # Assign here
        elif hasattr(model.features, 'denseblock3'): # Fallback for smaller custom densenets
            last_denseblock = model.features.denseblock3
            unfreeze_layers_names.append('denseblock3')
            last_denseblock_name = 'denseblock3' # Assign here
        elif hasattr(model.features, 'denseblock2'):
            last_denseblock = model.features.denseblock2
            unfreeze_layers_names.append('denseblock2')
            last_denseblock_name = 'denseblock2' # Assign here
        elif hasattr(model.features, 'denseblock1'):
            last_denseblock = model.features.denseblock1
            unfreeze_layers_names.append('denseblock1')
            last_denseblock_name = 'denseblock1' # Assign here

        # Try to find the transition layer before the last denseblock found
        # (e.g. transition3 if denseblock4 was found)
        last_transition = None
        if last_denseblock_name == 'denseblock4' and hasattr(model.features, 'transition3'):
            last_transition = model.features.transition3
            unfreeze_layers_names.append('transition3')
        elif last_denseblock_name == 'denseblock3' and hasattr(model.features, 'transition2'):
            last_transition = model.features.transition2
            unfreeze_layers_names.append('transition2')
        elif last_denseblock_name == 'denseblock2' and hasattr(model.features, 'transition1'):
            last_transition = model.features.transition1
            unfreeze_layers_names.append('transition1')
            
        final_norm = None
        if hasattr(model.features, 'norm5'): # Common for densenet121, 201
            final_norm = model.features.norm5
            unfreeze_layers_names.append('norm5')

        if last_denseblock is None:
            raise ValueError(
                "Strategy 'unfreeze_last_block' requires at least one 'denseblockX' (e.g., denseblock4, denseblock3) "
                "layer to be present in model.features. None were found. "
                "Please check your model architecture or use a different freeze_strategy."
            )

        # If we're here, last_denseblock is confirmed to be found.
        layers_to_unfreeze_objects = [last_denseblock]
        if last_transition:
            layers_to_unfreeze_objects.append(last_transition)
        if final_norm:
            layers_to_unfreeze_objects.append(final_norm)

  
        for layer_obj in layers_to_unfreeze_objects:
            for param in layer_obj.parameters():
                param.requires_grad = True
            layer_obj.train()           # <— restore train mode so BN & dropout work
        # …then ensure classifier is also in train
        model.classifier.train()
        print(f"Successfully unfroze layers: {sorted(list(set(unfreeze_layers_names)))} and classifier.")

        # Ensure classifier is trainable
        for param in model.classifier.parameters():
            param.requires_grad = True
            
    elif freeze_strategy == "unfreeze_all":
        for param in model.parameters():
            param.requires_grad = True
        print("Unfreezing all layers for full fine-tuning.")

    # --- discriminative learning rates ---
    head_lr     = 1e-3    # higher LR for the new classifier head
    backbone_lr = 1e-4    # lower LR for the pretrained backbone
    WEIGHT_DECAY = 1e-4

    # 1) set up optimizer with two param groups
    optimizer = optim.Adam([
        {"params": model.classifier.parameters(), "lr": head_lr},
        {"params": [p for p in model.features.parameters() if p.requires_grad],
         "lr": backbone_lr}
    ], weight_decay=WEIGHT_DECAY)

    # 2) use a cosine annealing schedule over your total epochs
    #    this will smoothly decay each group's LR toward zero
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,      # number of epochs for one full cosine cycle
        eta_min=1e-6       # optional: lower bound on LR
    )

    criterion = nn.MSELoss().to(DEVICE)
    
    trainer = Trainer(model, DEVICE, optimizer, criterion, save_dir=save_dir)
    print(f"Starting fine-tuning with strategy: {freeze_strategy if freeze_strategy else 'default'}")
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_trainable_params}")

    trainer.Loop(epochs, train_loader, val_loader, scheduler)
    
    # After training, evaluate on test set
    print("\nEvaluating model on test set...")
    # The best model is saved by the training Trainer in its save_dir
    best_model_path = os.path.join(save_dir, "best_model.pth")
    if not os.path.exists(best_model_path):
        # Fallback to the last model if best_model.pth isn't found (e.g. if saving best is disabled)
        # This part might need adjustment based on how your Trainer saves models
        # For now, we assume best_model.pth is the target.
        print(f"Warning: best_model.pth not found in {save_dir}. Evaluation might fail or use an unintended model.")
        # Potentially try to find latest checkpoint if applicable, or raise error
        # For now, we'll proceed, and `evaluate_model` will raise FileNotFoundError if `best_model_path` is bad.

    evaluate_model(
        checkpoint_path=best_model_path, # Use the saved model from the training phase
        visual_body_to_bmi_data=visual_body_to_bmi_data, # Pass the original dataframe
        absolute_path_col=absolute_path_col,
        large_model=large_model
    )

    # Example usage:
    
    # Make sure to have your data ready, e.g., face_data DataFrame
    # For example, load it as in train_densenet.py:
    # face_data_path = '/home/rajiv/rajiv-old/DigitalScale/data/filtered_datasets/filtered_data_december_with_face_only.csv'
    # if os.path.exists(face_data_path):
    #     face_data = pd.read_csv(face_data_path)
    #     face_data = face_data.dropna(subset=['face_only_photo_path'])
    #     face_data = face_data.drop(columns=['photo']) # Ensure correct path column is used
    # else:
    #     print(f"Warning: Example data file not found at {face_data_path}. Skipping example run.")
    #     face_data = None

    # --- STRATEGY 1: Fine-tune only the classifier ---
    # Assumes you have a pre-trained model (e.g., from train_densenet.py)
    # CHECKPOINT_PATH_LARGE = 'trained_models/face_only_large/best_model.pth' # Replace with your actual path
    # if face_data is not None and os.path.exists(CHECKPOINT_PATH_LARGE):
    #     print("\n--- Example: Fine-tuning only classifier (large model) ---")
    #     finetune_densenet(
    #         waybetter_data=face_data,
    #         save_dir='trained_models/finetune_face_large_classifier_only',
    #         epochs=10, # Adjust epochs for fine-tuning
    #         batch_size=32, # Can often use smaller batch size for fine-tuning
    #         absolute_path_col='face_only_photo_path',
    #         large_model=True,
    #         freeze_strategy="freeze_features",
    #         model_checkpoint_path=CHECKPOINT_PATH_LARGE
    #     )
    # else:
    #     print(f"Skipping example: 'Fine-tune only classifier' (check data and checkpoint: {CHECKPOINT_PATH_LARGE})")

    # --- STRATEGY 2: Fine-tune the last block and classifier ---
    # CHECKPOINT_PATH_REGULAR = 'trained_models/no_user_overlap_40_epochs/best_model.pth' # Replace with your actual path
    # db_path = "data/filtered_datasets/filtered_data_december.db"
    # if os.path.exists(db_path) and os.path.exists(CHECKPOINT_PATH_REGULAR):
    #     print("\n--- Example: Fine-tuning last block (regular model) ---")
    #     finetune_densenet(
    #         waybetter_data=db_path,
    #         save_dir='trained_models/finetune_regular_last_block',
    #         epochs=15,
    #         batch_size=32,
    #         large_model=False,
    #         freeze_strategy="unfreeze_last_block",
    #         model_checkpoint_path=CHECKPOINT_PATH_REGULAR
    #     )
    # else:
    #     print(f"Skipping example: 'Fine-tune last block' (check data and checkpoint: {CHECKPOINT_PATH_REGULAR})")


    # --- STRATEGY 3: Full fine-tuning (unfreeze all layers) ---
    # CHECKPOINT_PATH_LARGE_FULL = 'trained_models/face_only_large/best_model.pth' # Replace with your actual path
    # if face_data is not None and os.path.exists(CHECKPOINT_PATH_LARGE_FULL):
    #     print("\n--- Example: Full fine-tuning (large model) ---")
    #     finetune_densenet(
    #         waybetter_data=face_data,
    #         save_dir='trained_models/finetune_face_large_full',
    #         epochs=15,
    #         batch_size=16, # Potentially smaller batch for full fine-tuning if memory is an issue
    #         absolute_path_col='face_only_photo_path',
    #         large_model=True,
    #         freeze_strategy="unfreeze_all",
    #         model_checkpoint_path=CHECKPOINT_PATH_LARGE_FULL
    #     )
    # else:
    #     print(f"Skipping example: 'Full fine-tuning' (check data and checkpoint: {CHECKPOINT_PATH_LARGE_FULL})")
    
    # --- STRATEGY 4: Training from ImageNet, unfreezing last block ---
    # Useful if you don't have a domain-specific checkpoint but want to adapt ImageNet weights
    # This example assumes 'face_data' is loaded
    # if face_data is not None:
    #     print("\n--- Example: Training from ImageNet, unfreezing last block (large model) ---")
    #     finetune_densenet(
    #         waybetter_data=face_data,
    #         save_dir='trained_models/from_imagenet_face_large_unfreeze_last_block',
    #         epochs=20,
    #         batch_size=32,
    #         absolute_path_col='face_only_photo_path',
    #         large_model=True,
    #         freeze_strategy="unfreeze_last_block",
    #         model_checkpoint_path=None # Explicitly start from ImageNet
    #     )
    # else:
    #     print("Skipping example: 'Training from ImageNet, unfreezing last block' (check data)")
    
    print("\nScript finished. Uncomment and adapt examples in if __name__ == '__main__': to run fine-tuning.") 