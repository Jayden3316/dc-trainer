
import sys
import os
import torch

# Add project root to path
sys.path.append(os.getcwd())

from src.config.config import ConvNextEncoderConfig, ResNetEncoderConfig
from src.architecture.components.encoders import ConvNextEncoder, ResNetEncoder

def test_convnext_kernel_config():
    print("Testing ConvNextEncoder with custom downsample kernels...")
    
    # Define custom strides and kernels
    strides = [(2, 2), (2, 2), (2, 2)]
    kernels = [(3, 3), (3, 3), (3, 3)] # Kernel larger than stride
    
    cfg = ConvNextEncoderConfig(
        downsample_strides=strides,
        downsample_kernels=kernels
    )
    
    model = ConvNextEncoder(cfg)
    
    # Check layers
    print(f"Down2: stride={model.down2.conv.stride}, kernel={model.down2.conv.kernel_size}")
    print(f"Down3: stride={model.down3.conv.stride}, kernel={model.down3.conv.kernel_size}")
    print(f"Down4: stride={model.down4.conv.stride}, kernel={model.down4.conv.kernel_size}")
    
    assert model.down2.conv.kernel_size == (3, 3)
    assert model.down3.conv.kernel_size == (3, 3)
    assert model.down4.conv.kernel_size == (3, 3)
    
    # Test forward pass
    x = torch.randn(1, 3, 80, 200)
    out = model(x)
    print(f"Forward pass successful. Output shape: {out.shape}")

def test_convnext_default_kernel_config():
    print("\nTesting ConvNextEncoder with default downsample kernels (None)...")
    
    cfg = ConvNextEncoderConfig() # Default behavior
    model = ConvNextEncoder(cfg)
    
    # Check layers - should match default strides (2,2)
    print(f"Down2: stride={model.down2.conv.stride}, kernel={model.down2.conv.kernel_size}")
    
    assert model.down2.conv.kernel_size == (2, 2)
    
def test_resnet_kernel_config():
    print("\nTesting ResNetEncoder with custom downsample kernels...")
    
    strides = [(2, 2), (2, 2), (2, 2)]
    kernels = [(3, 3), (3, 3), (3, 3)]
    
    cfg = ResNetEncoderConfig(
        downsample_strides=strides,
        downsample_kernels=kernels
    )
    
    model = ResNetEncoder(cfg)
    
    print(f"Down2: stride={model.down2.conv.stride}, kernel={model.down2.conv.kernel_size}")
    assert model.down2.conv.kernel_size == (3, 3)
    
    x = torch.randn(1, 3, 80, 200)
    out = model(x)
    print(f"Forward pass successful. Output shape: {out.shape}")


if __name__ == "__main__":
    test_convnext_kernel_config()
    test_convnext_default_kernel_config()
    test_resnet_kernel_config()
    print("\nAll tests passed!")
