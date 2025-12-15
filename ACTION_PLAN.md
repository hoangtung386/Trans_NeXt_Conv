# 🎯 Action Plan - Hoàn Thiện Trans_NeXt_Conv Repository

## ✅ Các Lỗi Đã Được Sửa (Fixed)

### 1. Import Paths ✓
- [x] `configs/config.py`: Fixed `import set_determinism`
- [x] `src/training/trainer.py`: Fixed all imports
- [x] Script `fix_imports.py` để tự động sửa toàn bộ imports

### 2. Configuration ✓
- [x] Thêm validation cho config
- [x] Thêm environment variable support
- [x] Xử lý missing keys với defaults

### 3. Trainer Logic ✓
- [x] Fixed `validate()` method signature
- [x] Fixed batch unpacking từ dataloader
- [x] Fixed model output unpacking (5 values)
- [x] Fixed loss computation với auxiliary losses

### 4. Documentation ✓
- [x] README chi tiết với hướng dẫn setup
- [x] Troubleshooting guide
- [x] Training tips & hyperparameter tuning

---

## 🚧 Cần Làm Ngay (Priority: HIGH)

### 1. **Sửa Tất Cả Import Paths trong Project**
```bash
python fix_imports.py
```

**Files cần check manually:**
- `scripts/train.py`
- `scripts/evaluate.py` 
- `predict.py`
- `src/data/__init__.py`
- `src/visual_architeture/plot_architeture.py`

### 2. **Cập Nhật requirements.txt với Versions**
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
```

### 3. **Sửa evaluate.py**

**File:** `scripts/evaluate.py`

```python
# Line 41: Fixed model output unpacking
outputs, _, _, _, _ = self.model(images)  # Model returns 5 values
preds = torch.argmax(outputs, dim=1)
```

### 4. **Sửa predict.py**

```python
# Line 73: Fixed model output unpacking
with torch.no_grad():
    output, _, _, _, _ = model(image_tensor)  # Unpack 5 values
    pred_mask = torch.argmax(output, dim=1).cpu().numpy()[0]
```

### 5. **Sửa dataloader.py - Validation Transforms**

**File:** `src/data/dataloader.py`

```python
# Validation set NEEDS preprocessing!
val_ds = RSNA2DDataset(
    val_list, 
    transforms=True,  # ✅ Changed from False
    spatial_size=spatial_size
)
```

**Hoặc tốt hơn, refactor RSNA2DDataset:**

```python
class RSNA2DDataset(Dataset):
    def __init__(self, data_list, spatial_size=(256, 256), 
                 apply_preprocessing=True, apply_augmentation=False):
        """
        Args:
            apply_preprocessing: Resize, normalize (always True for train/val)
            apply_augmentation: Random crops, flips (only True for train)
        """
        self.apply_preprocessing = apply_preprocessing
        self.apply_augmentation = apply_augmentation
```

### 6. **Fix plot_architeture.py**

```python
# Line 1-10: Move device definition to top
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Then use device
model = get_model(CONFIG)
model = model.to(device)
model.eval()

dummy_input = torch.randn(1, 1, 256, 256).to(device)
```

---

## 🔨 Cần Làm Sau (Priority: MEDIUM)

### 7. **Thêm Data Validation**

**File mới:** `src/data/validate_data.py`

```python
def validate_data_structure(base_path):
    """Validate data directory structure and files"""
    required = {
        'dirs': ['train_images', 'segmentations'],
        'files': ['train_series_meta.csv', 'train_2024.csv']
    }
    
    errors = []
    
    for dir_name in required['dirs']:
        path = os.path.join(base_path, dir_name)
        if not os.path.exists(path):
            errors.append(f"Missing directory: {path}")
    
    for file_name in required['files']:
        path = os.path.join(base_path, file_name)
        if not os.path.exists(path):
            errors.append(f"Missing file: {path}")
    
    if errors:
        raise ValueError("\n".join(errors))
    
    print("✓ Data structure validated")

def validate_sample_data(base_path, num_samples=5):
    """Check if sample data can be loaded properly"""
    # Test loading logic here
    pass
```

### 8. **Thêm Error Handling**

**File:** `src/training/trainer.py`

```python
def train_epoch(self, epoch):
    try:
        # existing code
        pass
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("⚠️  CUDA out of memory! Try reducing batch_size")
            torch.cuda.empty_cache()
            raise
        else:
            raise

@torch.no_grad()
def validate(self):
    try:
        # existing code
        pass
    except Exception as e:
        print(f"❌ Validation error: {e}")
        raise
```

### 9. **Thêm Logging System**

**File mới:** `src/utils/logger.py`

```python
import logging
from datetime import datetime

def setup_logger(log_dir='./logs'):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)
```

### 10. **Thêm Unit Tests**

**File mới:** `tests/test_model.py`

```python
import pytest
import torch
from src.models.unet import TransNextConv

def test_model_forward():
    model = TransNextConv(
        image_size=256,
        n_channels=1,
        n_classes=6
    )
    
    x = torch.randn(2, 1, 256, 256)
    output, all_aux, aux1, aux2, aux3 = model(x)
    
    assert output.shape == (2, 6, 256, 256)
    assert isinstance(all_aux, list)

def test_model_shapes():
    # Test different input sizes
    pass
```

---

## 🎨 Cải Thiện Sau (Priority: LOW)

### 11. **Thêm Checkpoint Management**

```python
class CheckpointManager:
    def __init__(self, checkpoint_dir, max_checkpoints=5):
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
    
    def save(self, model, epoch, metrics):
        # Save and manage checkpoint rotation
        pass
    
    def load_best(self):
        # Load best checkpoint based on metrics
        pass
```

### 12. **Thêm TensorBoard Support**

```python
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, ...):
        self.writer = SummaryWriter(log_dir='./runs')
    
    def train_epoch(self, epoch):
        # Log to tensorboard
        self.writer.add_scalar('Loss/train', loss, epoch)
```

### 13. **Thêm Model Profiling**

```python
def profile_model(model, input_shape=(1, 1, 256, 256)):
    """Profile model memory and computation"""
    from torchinfo import summary
    
    stats = summary(
        model,
        input_size=input_shape,
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
        depth=4
    )
    
    return stats
```

### 14. **Tạo Docker Container**

**File mới:** `Dockerfile`

```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "scripts/train.py"]
```

### 15. **Thêm Pre-commit Hooks**

**File mới:** `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
  
  - repo: https://github.com/PyCQA/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
```

---

## 📝 Checklist Tổng Hợp

### Immediate Actions (Làm Ngay)
- [ ] Run `python fix_imports.py`
- [ ] Update `requirements.txt` với versions
- [ ] Fix `evaluate.py` model output unpacking
- [ ] Fix `predict.py` model output unpacking  
- [ ] Fix `dataloader.py` validation transforms
- [ ] Fix `plot_architeture.py` device definition
- [ ] Test training với `python scripts/train.py`

### Important (Quan Trọng)
- [ ] Add data validation functions
- [ ] Add error handling in trainer
- [ ] Setup logging system
- [ ] Write unit tests
- [ ] Create comprehensive documentation

### Nice to Have (Nên Có)
- [ ] Checkpoint management system
- [ ] TensorBoard integration
- [ ] Model profiling tools
- [ ] Docker support
- [ ] Pre-commit hooks

---

## 🧪 Testing Plan

### 1. Test Import Fix
```bash
python -c "from src.training.trainer import Trainer; print('✓ Imports OK')"
```

### 2. Test Config
```bash
python -c "from configs.config import CONFIG; print('✓ Config OK')"
```

### 3. Test Data Loading
```bash
python -c "
from src.data.dataloader import get_loaders
from configs.config import CONFIG
loaders = get_loaders(CONFIG)
print('✓ Data loading OK')
"
```

### 4. Test Model
```bash
python -c "
import torch
from src.models.initialize_model import get_model
from configs.config import CONFIG
model = get_model(CONFIG)
x = torch.randn(1, 1, 256, 256)
out = model(x)
print('✓ Model forward pass OK')
"
```

### 5. Test Training (Dry Run)
```bash
python scripts/train.py --num_epochs 1 --batch_size 2
```

---

## 📊 Estimated Timeline

| Phase | Tasks | Time | Status |
|-------|-------|------|--------|
| **Phase 1** | Fix imports, config, trainer | 1-2 hours | 🟡 In Progress |
| **Phase 2** | Data validation, error handling | 2-3 hours | ⚪ Not Started |
| **Phase 3** | Logging, tests | 3-4 hours | ⚪ Not Started |
| **Phase 4** | Improvements, Docker | 4-5 hours | ⚪ Not Started |

**Total Estimated Time:** 10-14 hours

---

## 💡 Tips

1. **Làm từng bước một** - Đừng cố sửa tất cả cùng lúc
2. **Test sau mỗi thay đổi** - Chạy test sau khi sửa mỗi file
3. **Commit thường xuyên** - Commit sau mỗi fix hoàn chỉnh
4. **Backup trước khi sửa** - Git branch hoặc copy files quan trọng

---

## 🆘 Nếu Gặp Vấn Đề

1. **Import errors** → Chạy `python fix_imports.py`
2. **CUDA OOM** → Giảm batch_size xuống 2-4
3. **Data not found** → Check `CONFIG['base_path']`
4. **Model errors** → Check model output unpacking (5 values)

---

**Last Updated:** December 2024
**Status:** 🟡 In Progress
