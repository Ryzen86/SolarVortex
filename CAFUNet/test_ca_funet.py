import torch
import time
from ca_funet import CAFUNet

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_ablation_staircase():
    print("==================================================")
    print("CA-FUNet Ablation Staircase Testing")
    print("==================================================")
    
    # User mentioned: 28 classes, individual cells input (1 channel usually, or 3 if RGB)
    # Using 1 channel and 28 classes for demonstration.
    in_channels = 1
    classes = 28
    input_tensor = torch.randn(2, in_channels, 256, 256)
    
    configs = [
        {"name": "Baseline (Standard U-Net)", "dual_path": False, "dynamic_weights": False},
        {"name": "+ Dynamic-Weighted Skips Only", "dual_path": False, "dynamic_weights": True},
        {"name": "+ Dual-Path Only (Fixed Weights)", "dual_path": True, "dynamic_weights": False},
        {"name": "CA-FUNet (Full)", "dual_path": True, "dynamic_weights": True},
    ]
    
    for config in configs:
        print(f"\n--- {config['name']} ---")
        
        # Initialize model
        model = CAFUNet(
            encoder_name='resnet34', 
            in_channels=in_channels, 
            classes=classes,
            use_dual_path=config["dual_path"],
            use_dynamic_weights=config["dynamic_weights"]
        )
        
        # Count parameters
        params = count_parameters(model)
        
        # Warmup for latency
        model.eval()
        with torch.no_grad():
            for _ in range(5):
                _ = model(input_tensor)
                
            # Measure latency
            start_time = time.time()
            for _ in range(20):
                output = model(input_tensor)
            end_time = time.time()
            
        latency_ms = ((end_time - start_time) / 20) * 1000
        
        print(f"Parameters: {params / 1e6:.2f} M")
        print(f"Latency (Batch Size 2, CPU): {latency_ms:.2f} ms")
        print(f"Output Shape: {output.shape} (Expected: [2, {classes}, 256, 256])")
        
        assert output.shape == (2, classes, 256, 256), f"Shape mismatch: {output.shape}"
        
    print("\nAll tests passed successfully!")

if __name__ == "__main__":
    test_ablation_staircase()
