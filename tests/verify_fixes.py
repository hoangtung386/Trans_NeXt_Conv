
import os
import sys
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.config import CONFIG
from src.models.unet import TransNextConv
from src.evaluation.evaluator import SegmentationEvaluator

def test_model_structure():
    print("Testing Model Structure...")
    # Update config for test
    CONFIG['spatial_size'] = [64, 64] # Small size for speed
    
    model = TransNextConv(
        image_size=64,
        n_channels=1,
        n_classes=6,
        embed_dim=256, # Smaller embed dim for test
        depths=[2, 2, 2, 2, 2, 2],
        widths=[64, 128, 256]
    )
    
    # Check experts count in decoder layers
    print(f"Decoder Layer 1 Experts: {model.decoder_layer_1.MoE.num_routed_experts} (Expected 16)")
    assert model.decoder_layer_1.MoE.num_routed_experts == 16
    
    # Test forward pass
    x = torch.randn(2, 1, 64, 64)
    print("Running forward pass...")
    try:
        output, _, aux1, aux2, aux3 = model(x)
        print("Forward pass successful!")
        print(f"Output shape: {output.shape}")
        assert output.shape == (2, 6, 64, 64)
    except Exception as e:
        print(f"Forward pass failed: {e}")
        raise e

def test_config_validation():
    print("\nTesting Config Validation...")
    from configs.config import validate_config
    # Should not crash even if path doesn't exist when check_data_path=False
    CONFIG['base_path'] = "/non/existent/path"
    validate_config(CONFIG, check_data_path=False)
    print("Config validation passed (dry run).")

def test_evaluator_unpacking():
    print("\nTesting Evaluator Logic (Mock Model)...")
    
    class MockModel(torch.nn.Module):
        def __call__(self, x):
            # Return tuple like the real model
            return torch.randn(x.shape[0], 6, 64, 64), [], {}, {}, {}
            
        def eval(self): pass
        def train(self): pass
    
    model = MockModel()
    
    # Mock data loader
    data = [{'image': torch.randn(1, 1, 64, 64), 'seg': torch.randint(0, 6, (1, 64, 64))}]
    val_loader = data
    
    evaluator = SegmentationEvaluator(model, val_loader, device='cpu', num_classes=6, output_dir='./temp_eval')
    
    # Run compute_metrics - should NOT crash
    try:
        evaluator.compute_metrics()
        print("Evaluator.compute_metrics() ran successfully (Crash fixed).")
    except Exception as e:
        print(f"Evaluator failed: {e}")
        raise e

if __name__ == "__main__":
    try:
        test_model_structure()
        test_config_validation()
        test_evaluator_unpacking()
        print("\nAll verification tests passed!")
    except Exception as e:
        print(f"\nVerification FAILED: {e}")
        sys.exit(1)