import torch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.unet import TransNextConv
from configs.config import CONFIG

def test_model_forward():
    print("Initializing model...")
    device = torch.device('cpu') # Use CPU for quick test
    model = TransNextConv(
        image_size=256,
        n_channels=1,
        n_classes=6,
        embed_dim=128, # Reduced for quick test
        widths=[32, 64, 128], # Reduced
        depths=[2, 2, 2, 2, 2, 2]
    ).to(device)
    
    batch_size = 2
    x = torch.randn(batch_size, 1, 256, 256).to(device)
    
    print("Running forward pass...")
    try:
        output, all_aux_losses, aux_loss_1, aux_loss_2, aux_loss_3 = model(x)
        print("Forward pass successful.")
        
        print(f"Output shape: {output.shape}")
        expected_shape = (batch_size, 6, 256, 256)
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
        
        print("Checking Aux Losses...")
        # Check structure of aux losses
        assert isinstance(all_aux_losses, list)
        assert isinstance(aux_loss_1, dict)
        assert 'len(all_aux_losses)'
        
        # Check keys in aux_loss_1
        expected_keys = ['load_balance_loss', 'router_entropy', 'shared_routed_balance_loss']
        for k in expected_keys:
            assert k in aux_loss_1, f"Missing key {k} in aux_loss_1"
            val = aux_loss_1[k]
            assert isinstance(val, torch.Tensor) or isinstance(val, float), f"Value for {k} should be tensor or float"
            if isinstance(val, torch.Tensor):
                 assert val.numel() == 1
        
        print("Aux losses structure verified.")
        print("Verification PASSED!")
        
    except Exception as e:
        print(f"Verification FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_model_forward()
    