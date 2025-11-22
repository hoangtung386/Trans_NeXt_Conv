"""
Quantization utilities for TransNextConv model
Optimized for GPU
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler


class ModelQuantizer:
    """Utility class for model quantization and mixed precision training"""

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.scaler = GradScaler()

    def convert_to_fp16(self, model=None):
        """
        Convert model to FP16 (mixed precision) manually.
        Keeps BatchNorm and LayerNorm in FP32 for stability.
        """
        # Nếu không truyền model vào thì dùng self.model
        if model is None:
            model = self.model

        for name, child in model.named_children():
            # Giữ normalization layers ở định dạng FP32
            if isinstance(child, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                continue
            # Giữ embedding layers ở định dạng FP32
            elif isinstance(child, nn.Embedding):
                continue
            # Convert Linear and Conv layers sang FP16
            elif isinstance(child, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                child.half()
            else:
                self.convert_to_fp16(child)

        return model

    def prepare_for_training(self, optimizer):
        """
        Prepare model and optimizer for mixed precision training
        """
        # Convert model to FP16 (Manual conversion)
        self.convert_to_fp16()

        # Di chuyển model sang device trước khi đưa vào optimizer
        self.model = self.model.to(self.device)

        print("Đã bật huấn luyện độ chính xác hỗn hợp")
        print("Sử dụng GradScaler để điều chỉnh tỷ lệ gradient")

        return self.model, optimizer, self.scaler

    def training_step(self, model, data, target, optimizer, criterion, scaler):
        """
        Execute one training step with mixed precision
        """
        model = model.to(self.device)
        data = data.to(self.device)
        target = target.to(self.device)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast():
            output = model(data)
            loss = criterion(output, target)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        return loss.item()

    @torch.no_grad()
    def validation_step(self, model, data):
        """
        Execute validation step with mixed precision
        """
        model = model.to(self.device)
        data = data.to(self.device)

        with autocast():
            output = model(data)
        return output
