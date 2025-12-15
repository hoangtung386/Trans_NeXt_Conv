# Trans_NeXt_Conv Refactoring Walkthrough

I have successfully refactored the codebase to resolve the critical structural errors, import cycles, and logic mismatches. The project is now structured to support proper training and inference workflows.

## Changes Validation

### 1. Structural Fixes
- **Global Execution Removed**: [src/models/initialize_model.py](file:///home/admin1/Projects/Trans_NeXt_Conv/src/models/initialize_model.py), [src/data/dataset.py](file:///home/admin1/Projects/Trans_NeXt_Conv/src/data/dataset.py), and [src/data/dataloader.py](file:///home/admin1/Projects/Trans_NeXt_Conv/src/data/dataloader.py) no longer execute code on import. They now provide factory functions:
    - [get_model(config)](file:///home/admin1/Projects/Trans_NeXt_Conv/src/models/initialize_model.py#7-56)
    - [prepare_data(config)](file:///home/admin1/Projects/Trans_NeXt_Conv/src/data/dataset.py#9-85)
    - [get_loaders(config)](file:///home/admin1/Projects/Trans_NeXt_Conv/src/data/dataloader.py#8-64)
- **[scripts/train.py](file:///home/admin1/Projects/Trans_NeXt_Conv/scripts/train.py) Updated**: Now correctly imports these factory functions and initializes the model/loaders inside the [main](file:///home/admin1/Projects/Trans_NeXt_Conv/predict.py#58-102) block.
- **[scripts/evaluate.py](file:///home/admin1/Projects/Trans_NeXt_Conv/scripts/evaluate.py) Updated**: Similar fixes applied to prevent side-effects during import.

### 2. Logic Corrections
- **Trainer Class**: Duplicate [train_epoch](file:///home/admin1/Projects/Trans_NeXt_Conv/src/training/trainer.py#125-199) method removed. The remaining method handles Mixed Precision (AMP) and correctly aggregates the multiple auxiliary losses from the model.
- **Model Output Sync**: Confirmed [TransNextConv](file:///home/admin1/Projects/Trans_NeXt_Conv/src/models/unet.py#21-203) returns 5 values. [Trainer](file:///home/admin1/Projects/Trans_NeXt_Conv/src/training/trainer.py#17-337) now correctly unpacks them (`output, all_aux, aux1, aux2, aux3`) and passes them to [CombinedLoss](file:///home/admin1/Projects/Trans_NeXt_Conv/src/losses/combined_loss.py#7-79).
- **CombinedLoss**: [Trainer](file:///home/admin1/Projects/Trans_NeXt_Conv/src/training/trainer.py#17-337) bundles the auxiliary losses into a list to match [CombinedLoss](file:///home/admin1/Projects/Trans_NeXt_Conv/src/losses/combined_loss.py#7-79) signature.

### 3. cleanup & Optimization
- **Circular Import Fixed**: [src/visual_architeture/plot_architeture.py](file:///home/admin1/Projects/Trans_NeXt_Conv/src/visual_architeture/plot_architeture.py) no longer imports from [train.py](file:///home/admin1/Projects/Trans_NeXt_Conv/scripts/train.py). It instantiates its own model for visualization.
- **Requirements**: [requirements.txt](file:///home/admin1/Projects/Trans_NeXt_Conv/requirements.txt) updated with missing libraries (monai, wandb, etc.).

## How to Run

### Training
Ensure your environment is set up with dependencies in [requirements.txt](file:///home/admin1/Projects/Trans_NeXt_Conv/requirements.txt).
```bash
python scripts/train.py --num_epochs 100 --batch_size 16
```

### Inference
I added a new script [predict.py](file:///home/admin1/Projects/Trans_NeXt_Conv/predict.py) for single-image inference.
```bash
python predict.py --image_path /path/to/image.nii --checkpoint output/best_model.pth
```

### Visualization
To visualize architecture without running training:
```bash
python src/visual_architeture/plot_architeture.py
```
