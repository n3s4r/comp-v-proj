import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class SKResNeXtUNet(nn.Module):
    def __init__(self, in_channels=4, classes=24):
        super(SKResNeXtUNet, self).__init__()
        
        # Using segmentation_models_pytorch with timm encoder
        # Encoder name for SK-ResNeXt-50 might be 'skresnext50_32x4d' in timm
        # We need to ensure the weights are available or use 'imagenet'
        
        self.model = smp.Unet(
            encoder_name="skresnext50_32x4d", # Check if this specific name is supported in current SMP version
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=classes,
        )

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    # Test the model instantiation
    model = SKResNeXtUNet()
    x = torch.randn(1, 4, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

