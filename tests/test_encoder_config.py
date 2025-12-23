
import torch
from src.config.config import AsymmetricConvNextEncoderConfig, ResNetEncoderConfig
from src.architecture.components.encoders import AsymmetricConvNextEncoder, ResNetEncoder

def test_encoders():
    print("Testing AsymmetricConvNextEncoder...")
    # Test default (2, 1) strides
    cfg = AsymmetricConvNextEncoderConfig()
    print(f"Default strides: {cfg.downsample_strides}")
    model = AsymmetricConvNextEncoder(cfg)
    x = torch.randn(1, 3, 80, 200)
    out = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}") 
    # Height should be 80 -> 20 (stem /4) -> 10 (down2) -> 5 (down3) -> 2 (down4)
    # Width should be 200 -> 50 (stem /4) -> 50 -> 50 -> 50
    # Expected: [1, 512, 2, 50]
    assert out.shape == (1, 512, 2, 50)
    print("Default config passed.")

    # Test custom strides (aggressive width reduction)
    cfg_custom = AsymmetricConvNextEncoderConfig(
        downsample_strides=[(1, 2), (1, 2), (1, 2)]
    )
    print(f"Custom strides: {cfg_custom.downsample_strides}")
    model_custom = AsymmetricConvNextEncoder(cfg_custom)
    out_custom = model_custom(x)
    print(f"Input: {x.shape}, Output: {out_custom.shape}")
    # Height: 80 -> 20 (stem) -> 20 -> 20 -> 20 (no vertical downsample)
    # Width: 200 -> 50 (stem) -> 25 -> 12 -> 6
    # Expected: [1, 512, 20, 6]
    assert out_custom.shape == (1, 512, 20, 6)
    print("Custom config passed.")


    print("\nTesting ResNetEncoder...")
    # Test default (2, 2) strides
    cfg_resnet = ResNetEncoderConfig()
    print(f"Default strides: {cfg_resnet.downsample_strides}")
    model_resnet = ResNetEncoder(cfg_resnet)
    out_resnet = model_resnet(x)
    print(f"Input: {x.shape}, Output: {out_resnet.shape}")
    # Height: 80 -> 20 (stem /4) -> 10 -> 5 -> 2
    # Width: 200 -> 50 (stem /4) -> 25 -> 12 -> 6
    # Expected: [1, 512, 2, 6]
    assert out_resnet.shape == (1, 512, 2, 6)
    print("Default ResNet passed.")

    # Test mixed strides
    cfg_resnet_custom = ResNetEncoderConfig(
        downsample_strides=[(2, 1), (1, 2), (2, 2)]
    )
    print(f"Custom ResNet strides: {cfg_resnet_custom.downsample_strides}")
    model_resnet_custom = ResNetEncoder(cfg_resnet_custom)
    out_resnet_custom = model_resnet_custom(x)
    print(f"Input: {x.shape}, Output: {out_resnet_custom.shape}")
    # Stem: 20x50
    # Down2 (2,1): 10x50
    # Down3 (1,2): 10x25
    # Down4 (2,2): 5x12
    # Expected: [1, 512, 5, 12]
    # Note: 25 / 2 = 12 (floor)
    assert out_resnet_custom.shape == (1, 512, 5, 12)
    print("Custom ResNet passed.")

if __name__ == "__main__":
    test_encoders()
