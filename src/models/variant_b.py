import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class VariantB(nn.Module):
    """
    Variant B – SK-ResNeXt Encoder + Dense (U-Net++) Decoder
    
    Backbone: SK-ResNeXt-50 (ImageNet-pretrained)
    Decoder: Dense (U-Net++)-style decoder with deep supervision
    """
    def __init__(self, in_channels=4, classes=24, deep_supervision=True):
        super(VariantB, self).__init__()
        
        self.deep_supervision = deep_supervision
        
        # U-Net++ implementation from segmentation_models_pytorch
        self.model = smp.UnetPlusPlus(
            encoder_name="skresnext50_32x4d",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=classes,
            decoder_use_batchnorm=True,
            deep_supervision=deep_supervision,
        )

    def forward(self, x):
        return self.model(x)

class DeepSupervisionLoss(nn.Module):
    """
    Deep Supervision Loss: Multi-output Dice + Cross-Entropy Loss
    """
    def __init__(self, weights=None, ignore_index=None):
        super(DeepSupervisionLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index if ignore_index is not None else -100)
        self.dice_loss = smp.losses.DiceLoss(mode='multiclass', ignore_index=ignore_index)

    def forward(self, outputs, target):
        """
        Args:
            outputs: List of tensors if deep_supervision is True, else single tensor
            target: Ground truth mask
        """
        loss = 0
        
        if isinstance(outputs, (list, tuple)):
            # Deep supervision: outputs is a list of tensors from different depths
            # Typically, we assign weights to different scales or average them
            # Here we simply sum them up as is common in deep supervision, 
            # or we could weight the final output higher.
            for output in outputs:
                ce = self.ce_loss(output, target)
                dice = self.dice_loss(output, target)
                loss += (ce + dice)
            
            # Average over number of outputs to keep loss scale similar?
            # Or just sum. Let's average for stability.
            loss /= len(outputs)
        else:
            ce = self.ce_loss(outputs, target)
            dice = self.dice_loss(outputs, target)
            loss = ce + dice
            
        return loss

if __name__ == "__main__":
    # Test the model instantiation
    model = VariantB(deep_supervision=True)
    x = torch.randn(2, 4, 256, 256) # Batch size 2
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    if isinstance(y, list):
        print(f"Output (Deep Supervision): List of {len(y)} tensors")
        for i, out in enumerate(y):
            print(f"  Output {i} shape: {out.shape}")
    else:
        print(f"Output shape: {y.shape}")
        
    # Test Loss
    loss_fn = DeepSupervisionLoss()
    target = torch.randint(0, 24, (2, 256, 256)).long()
    loss = loss_fn(y, target)
    print(f"Loss value: {loss.item()}")

