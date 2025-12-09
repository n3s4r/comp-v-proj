# Spatio-Temporal Adaptive Fusion Transformer (STAFT)

**Department of Electrical & Computer Engineering, North South University**
**Course:** CSE 468

## Group Members
- Khan Mohammad Sashoto Seeam (1831769642)
- Nesar Ahmed (2211836642)
- Ahnaf Mohammed Mahi Kabir (2222171042)

**Supervisor:** Dr. Md Adnan Arefeen [AFE]

## Introduction
Land Cover Classification (LCC) identifies surface types (water, vegetation, urban areas) from satellite images. While U-Net and DeepLab are effective, they struggle with complex patterns in multispectral images. This project aims to improve pixel-wise land-cover segmentation accuracy and efficiency by enhancing U-Net with **Selective Kernel ResNeXt (SK-ResNeXt)** modules and exploring Transformer-based fusion.

## Project Goals
1.  **Reproduce Baseline:** U-Net with SK-ResNeXt-50 encoder.
2.  **Variant A:** Add Attention Gates to the decoder.
3.  **Variant B:** Use a Dense (U-Net++)-style decoder.
4.  **Variant C (Hybrid):** Parallel SK-ResNeXt-50 and ViT-Tiny/SwinT encoders with attention-based fusion.

## Dataset
**Five-Billion-Pixels Dataset:**
- 150 high-resolution Gaofen-2 satellite images.
- 4 spectral bands: Blue, Green, Red, Near-Infrared (NIR).
- 24 land cover categories.

## Evaluation Metrics
- **Overall Accuracy (OA)**
- **Intersection over Union (IoU)**
- **Mean Intersection over Union (mIoU)**
- **Training/Inference Time**

## Implementation Details
- **Framework:** PyTorch
- **Input:** 256x256 pixel patches
- **Optimizer:** Adam / AdamW
- **Loss Functions:** Cross-Entropy, Dice Loss, Weighted Cross-Entropy

### Baseline (U-Net + SK-ResNeXt-50)
- Backbone: SK-ResNeXt-50
- Loss: Cross-Entropy
- Augmentation: Random flips & rotations

### Variants
- **Variant A:** Attention Gates, Weighted CE + Dice Loss.
- **Variant B:** Dense connections (U-Net++ style), Deep Supervision.
- **Variant C:** Hybrid CNN + ViT Fusion.

