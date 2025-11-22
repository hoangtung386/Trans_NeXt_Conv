# Creater file

### Hướng dẫn 
```
# Chạy với config mặc định
python scripts/train.py

# Thay đổi batch size và learning rate
python scripts/train.py --batch_size 32 --lr 1e-3

# Thay đổi nhiều tham số
python scripts/train.py --batch_size 16 --num_epochs 50 --lr 5e-4 --output_dir ./exp1

# Tắt W&B logging
python scripts/train.py --no_wandb

# Chạy trên CPU
python scripts/train.py --device cpu

# Thay đổi spatial size
python scripts/train.py --spatial_size 512 512

# Xem tất cả options
python scripts/train.py --help
```