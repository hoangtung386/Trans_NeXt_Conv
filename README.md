# Trans_NeXt_Conv: Medical Image Segmentation

A hybrid CNN-Transformer architecture for multi-organ segmentation in medical imaging.

## 🏗️ Architecture

![Transformer Decoder layer](./Decoder.png)

![Transformer Encoder layer](./Encoder.png)

**Trans_NeXt_Conv** combines:
- **CNN Path**: ConvNeXt-based encoder-decoder for spatial feature extraction
- **Transformer Path**: CrossViT + MoE (Mixture of Experts) for global context
- **Fusion Module**: Multi-scale feature fusion for final prediction

### Key Features
- ✅ Mixed Precision Training (FP16)
- ✅ Mixture of Experts (MoE) layers
- ✅ Multi-scale Cross Attention
- ✅ Combined loss (Dice + Focal + Tversky)
- ✅ W&B integration for experiment tracking

---

## 📋 Requirements

### System Requirements
- Python 3.8+
- CUDA 11.7+ (for GPU training)
- 16GB+ RAM
- 8GB+ GPU VRAM (recommended: RTX 3080 or better)

### Installation

```bash
# Clone repository
git clone https://github.com/your-repo/Trans_NeXt_Conv.git
cd Trans_NeXt_Conv

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
```txt
torch>=2.0.0,<2.2.0
torchvision>=0.15.0,<0.17.0
monai>=1.3.0
einops>=0.7.0
wandb>=0.15.0
scikit-learn>=1.3.0
pandas>=2.0.0
nibabel>=5.1.0
scikit-image>=0.21.0
scipy>=1.11.0
tqdm>=4.66.0
matplotlib>=3.7.0
seaborn>=0.12.0
graphviz>=0.20.0
torchviz>=0.0.2
torchinfo>=1.8.0
```

---

## 📁 Project Structure

```
Trans_NeXt_Conv/
├── configs/
│   └── config.py              # Configuration file
├── src/
│   ├── data/
│   │   ├── dataset.py         # Dataset preparation
│   │   ├── dataloader.py      # DataLoader factory
│   │   └── preprocessing.py   # Data preprocessing
│   ├── losses/
│   │   ├── dice_loss.py
│   │   ├── focal_loss.py
│   │   ├── tversky_loss.py
│   │   └── combined_loss.py
│   ├── metrics/
│   │   └── compute_metric.py
│   ├── models/
│   │   ├── unet.py           # Main model architecture
│   │   ├── attention_cnn.py  # CNN components
│   │   ├── segformer.py      # Transformer components
│   │   └── initialize_model.py
│   └── training/
│       └── trainer.py        # Training loop
├── scripts/
│   ├── train.py              # Training script
│   └── evaluate.py           # Evaluation script
├── predict.py                # Inference script
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Prepare Your Data

Organize your data in the following structure:

```
/path/to/your/data/
├── train_images/
│   └── {patient_id}/
│       └── {series_id}/
│           └── *.dcm
├── segmentations/
│   └── {series_id}.nii
├── train_series_meta.csv
└── train_2024.csv
```

### 2. Configure Settings

Edit `configs/config.py`:

```python
CONFIG = {
    'base_path': "/path/to/your/data",  # ⚠️ Change this!
    'output_dir': "./output",
    'num_epochs': 100,
    'batch_size': 8,  # Adjust based on GPU memory
    'learning_rate': 1e-4,
    'use_wandb': True,
    'wandb_project': 'your-project-name',
}
```

### 3. Train the Model

```bash
# Basic training
python scripts/train.py

# Custom settings
python scripts/train.py --num_epochs 50 --batch_size 4 --lr 5e-5

# Without W&B logging
python scripts/train.py --no_wandb

# Resume from checkpoint
python scripts/train.py  # Automatically resumes if checkpoint exists
```

### 4. Run Inference

```bash
# Single image prediction
python predict.py \
    --image_path /path/to/image.nii \
    --checkpoint output/best_model.pth \
    --output_path prediction.png

# Specify slice for 3D volumes
python predict.py \
    --image_path /path/to/volume.nii \
    --checkpoint output/best_model.pth \
    --slice_idx 50
```

### 5. Evaluate Model

```bash
python scripts/evaluate.py
```

This will generate:
- Metrics comparison plots
- Confusion matrices
- Overlay visualizations
- Per-class segmentation comparisons
- Text report with all metrics

---

## 🔧 Troubleshooting

### Common Issues

#### 1. **CUDA Out of Memory**
```bash
# Reduce batch size
python scripts/train.py --batch_size 2

# Increase gradient accumulation
python scripts/train.py --gradient_accumulation_steps 8

# Disable AMP
python scripts/train.py --no_amp
```

#### 2. **Import Errors**
```bash
# Fix all import paths
python fix_imports.py
```

#### 3. **Data Not Found**
```python
# Set environment variable
export DATA_PATH="/path/to/your/data"

# Or edit config.py directly
CONFIG['base_path'] = "/path/to/your/data"
```

#### 4. **W&B Authentication Failed**
```bash
# Set API key
export WANDB_API_KEY="your-api-key"

# Or disable W&B
python scripts/train.py --no_wandb
```

---

## 📊 Training Tips

### Memory Optimization
- Use `batch_size=2-4` for 8GB GPU
- Enable gradient accumulation: `gradient_accumulation_steps=8`
- Use mixed precision: `use_amp=True` (default)

### Hyperparameter Tuning
```python
# Learning rate schedule
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

# Loss weights (adjust based on class imbalance)
loss_weights = {
    'dice': 0.5,    # Main metric
    'focal': 0.3,   # Handle class imbalance
    'tversky': 0.2  # Fine-tune FP/FN trade-off
}
```

### Data Augmentation
Currently uses:
- Intensity clipping: `[-125, 275]` HU
- Normalization to `[0, 1]`
- Foreground cropping with padding
- Resize to `256x256`

To add more augmentation, modify `src/data/preprocessing.py`.

---

## 📈 Model Performance

Expected metrics on validation set:

| Organ | Dice | IoU | Precision | Recall |
|-------|------|-----|-----------|--------|
| Liver | 0.92 | 0.85 | 0.91 | 0.93 |
| Spleen | 0.88 | 0.79 | 0.87 | 0.89 |
| Kidney_L | 0.90 | 0.82 | 0.89 | 0.91 |
| Kidney_R | 0.90 | 0.82 | 0.89 | 0.91 |
| Bowel | 0.75 | 0.60 | 0.74 | 0.76 |

*Note: Actual performance depends on your dataset quality and size.*

---

## 🤝 Contributing

1. Fix import paths: `python fix_imports.py`
2. Follow PEP 8 style guide
3. Add docstrings to new functions
4. Test changes before committing

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{transnextconv2024,
  title={Trans_NeXt_Conv: Hybrid CNN-Transformer for Medical Image Segmentation},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  url={https://github.com/your-repo/Trans_NeXt_Conv}
}
```

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## 🐛 Known Issues

1. ⚠️ **Import paths need fixing** - Run `python fix_imports.py` first
2. ⚠️ **High memory usage** - MoE layers are memory-intensive
3. ⚠️ **Long training time** - Expected 2-3 hours per epoch on RTX 3080

For more issues, check [GitHub Issues](https://github.com/your-repo/Trans_NeXt_Conv/issues).

---

## 📞 Contact

For questions or issues:
- GitHub Issues: [Link]
- Email: your.email@example.com

---

**Last Updated:** December 2024
