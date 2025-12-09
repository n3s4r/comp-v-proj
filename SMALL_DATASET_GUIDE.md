## Training on a Small Subset (10 Images)

If you want to quickly test the pipeline using just 10 images from the **Image__8bit_NirRGB** and **Annotation__index** folders, follow these steps.

### Step 1: Download & Organize Data
1.  Download 10 matching files from `Image__8bit_NirRGB` (images) and `Annotation__index` (masks) from the Google Drive link.
    *   *Note: Ensure the filenames match exactly (e.g., `123.tif` in images and `123.tif` in masks).*
2.  Upload them to your Google Drive in a specific structure:

```
/content/drive/MyDrive/STAFT_Project/
  └── small_data/
      └── train/
          ├── images/  <-- Put your 10 .tif images here
          └── masks/   <-- Put your 10 .tif masks here
```

### Step 2: Update `LandCoverDataset`
In your Colab notebook (Cell 3, `src/dataset.py`), update the file loading logic to point to this new folder and handle the 8-bit format.

```python
%%writefile src/dataset.py
import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import glob

class LandCoverDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Point to the specific folders
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, split, 'images', '*.tif')))
        self.mask_paths = sorted(glob.glob(os.path.join(root_dir, split, 'masks', '*.tif')))
        
        print(f"Found {len(self.image_paths)} images and {len(self.mask_paths)} masks")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # LOAD IMAGES (8-bit NirRGB)
        # 8-bit images can be loaded directly. 
        # Note: 'Image__8bit_NirRGB' usually has 4 channels (NIR, R, G, B) or (R, G, B, NIR).
        # We need to ensure we read it correctly.
        # cv2.imread usually reads BGR. For multi-channel TIFF, use IMREAD_UNCHANGED.
        
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        
        # LOAD MASKS (Annotation__index)
        # These are likely grayscale where pixel value = class index (0-23)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        # Basic Check
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # Ensure mask is 2D [H, W]
        if len(mask.shape) > 2:
            mask = mask[:, :, 0] # Take first channel if it happened to be saved as 3-channel

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # To Tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() # [C, H, W]
            # Normalize 8-bit (0-255) to 0-1 range for stability
            image = image / 255.0 
            
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()

        return image, mask
```

### Step 3: Run Training with Small Batch
Since you only have 10 images, use a smaller batch size (e.g., 2) and run for more epochs (e.g., 50) to see if it overfits (loss goes to 0).

In your `train.py` or Notebook training cell:
```python
# ... inside main() ...
dataset = LandCoverDataset(root_dir="/content/drive/MyDrive/STAFT_Project/small_data")

# Use a small batch size
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Run training...
```

