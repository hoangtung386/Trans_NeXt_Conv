import torch 
from src.models.unet import TransNextConv
from src.models.unet import quantization
from torch.cuda.amp import autocast, GradScaler
from quantization import ModelQuantizer
import gc
from configs.config import CONFIG

if __name__ == "__main__":
    batch_size = CONFIG.batch_size
    image_size = CONFIG.image_crop_size

    # Dọn dẹp bộ nhớ trước khi bắt đầu
    gc.collect()
    torch.cuda.empty_cache()

    n_channels = CONFIG.n_channels
    n_classes = CONFIG.n_classes

    # Create model
    model = TransNextConv(
        image_size=image_size,
        n_channels=n_channels,
        n_classes=n_classes,
        stem_features=64,
        depths=[3, 4, 6, 2, 2, 2],
        widths=[256, 512, 1024],
        drop_p=0.0,
        embed_dim=1024
    )
    # Quantization
    quantizer = ModelQuantizer(model, CONFIG.device)

    # Lưu ý: Hàm này sẽ biến đổi trực tiếp model in-place
    model_fp16 = quantizer.convert_to_fp16(model)
    model_fp16 = model.to(CONFIG.device)

    # Create random input
    # SỬA LỖI 1: Thêm .half() để chuyển input sang FP16 khớp với model
    x = torch.randn(batch_size, n_channels, image_size, image_size).to(CONFIG.device).half()

    # Forward pass
    try:
        with torch.no_grad():
            # SỬA LỖI 2: Bọc trong autocast để xử lý các layer hỗn hợp (Conv FP16 -> BN FP32)
            with autocast():
                output = model(x)

        print(f"\nForward pass successful!")
        print(f"Input shape:  {x.shape}")
        print(f"Output shape: {output.shape}")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # ... (Phần code in kiến trúc bên dưới giữ nguyên) ...
        print("\n" + "-"*60)
        print("MODEL ARCHITECTURE OVERVIEW")
        print("-"*60)
        # ...

    except Exception as e:
        print(f"\nError during forward pass:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


    # Forward pass
    try:
        with torch.no_grad():
            output = model(x)
        print(f"\nForward pass successful!")
        print(f"Input shape:  {x.shape}")
        print(f"Output shape: {output.shape}")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        print("\n" + "-"*60)
        print("MODEL ARCHITECTURE OVERVIEW")
        print("-"*60)
        print("\nCNN Encoder Path:")
        print(f"  in_conv:  [{n_channels}, {image_size}, {image_size}] → [64, {image_size//4}, {image_size//4}]")
        print(f"  enc_1:    [64, {image_size//4}, {image_size//4}] → [256, {image_size//8}, {image_size//8}]")
        print(f"  enc_2:    [256, {image_size//8}, {image_size//8}] → [512, {image_size//16}, {image_size//16}]")
        print(f"  enc_3:    [512, {image_size//16}, {image_size//16}] → [1024, {image_size//32}, {image_size//32}]")

        print("\nTransformer Path:")
        print(f"  Encoder:  Image → Tokens → Transformer Features")
        print(f"  Decoder:  3 layers with cross-attention and MoE")

        print("\nCNN Decoder Path:")
        print(f"  dec_1:    [1024, {image_size//32}, {image_size//32}] → [512, {image_size//16}, {image_size//16}]")
        print(f"  dec_2:    [512, {image_size//16}, {image_size//16}] → [256, {image_size//8}, {image_size//8}]")
        print(f"  dec_3:    [256, {image_size//8}, {image_size//8}] → [64, {image_size//4}, {image_size//4}]")
        print(f"  upsample: [64, {image_size//4}, {image_size//4}] → [64, {image_size}, {image_size}]")

        print("\nFusion & Output:")
        print(f"  fusion:   [128, {image_size}, {image_size}] → [64, {image_size}, {image_size}]")
        print(f"  output:   [64, {image_size}, {image_size}] → [{n_classes}, {image_size}, {image_size}]")

    except Exception as e:
        print(f"\nError during forward pass:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
