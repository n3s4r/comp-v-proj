# Training STAFT Variant B on Google Colab

This guide will help you train the **Variant B (SK-ResNeXt + U-Net++)** model on Google Colab using the Free GPU tier.

## Step 1: Prepare Your Data (Five-Billion-Pixels Dataset)
Since the dataset is large, you should upload it to **Google Drive** for easy access.
1.  Download the dataset (if you haven't already).
2.  Upload the `train` and `val` folders to your Google Drive (e.g., in a folder named `STAFT_Project/data`).
3.  *Optional but Recommended:* Zip the folders before uploading to upload faster, then unzip them in Colab.

## Step 2: Open Google Colab
1.  Go to [Google Colab](https://colab.research.google.com/).
2.  Create a **New Notebook**.
3.  Go to **Runtime > Change runtime type**.
4.  Select **T4 GPU** (or better if you have Colab Pro).

## Step 3: Setup the Environment
Copy and paste the following code blocks into cells in your Colab notebook.

### Cell 1: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 2: Install Dependencies
```python
!pip install segmentation-models-pytorch albumentations
```

### Cell 3: Setup Project Files
We will create the necessary python files directly in Colab.

**Create `src/models/variant_b.py`:**
```python
import os
os.makedirs('src/models', exist_ok=True)

%%writefile src/models/variant_b.py
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class VariantB(nn.Module):
    def __init__(self, in_channels=4, classes=24, deep_supervision=True):
        super(VariantB, self).__init__()
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
    def __init__(self, weights=None, ignore_index=None):
        super(DeepSupervisionLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index if ignore_index is not None else -100)
        self.dice_loss = smp.losses.DiceLoss(mode='multiclass', ignore_index=ignore_index)

    def forward(self, outputs, target):
        loss = 0
        if isinstance(outputs, (list, tuple)):
            for output in outputs:
                ce = self.ce_loss(output, target)
                dice = self.dice_loss(output, target)
                loss += (ce + dice)
            loss /= len(outputs)
        else:
            loss = self.ce_loss(outputs, target) + self.dice_loss(outputs, target)
        return loss
```

**Create `src/dataset.py`:**
*Note: You need to update the file loading logic here to match your specific dataset folder structure on Drive.*

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
        
        # Example: looking for .tif files. Update pattern as needed.
        # self.image_paths = sorted(glob.glob(os.path.join(root_dir, split, 'images', '*.tif')))
        # self.mask_paths = sorted(glob.glob(os.path.join(root_dir, split, 'masks', '*.tif')))
        
        # Placeholder for testing if you don't have data yet
        self.image_paths = ["dummy"] * 100 
        self.mask_paths = ["dummy"] * 100

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # REAL LOADING LOGIC (Uncomment and adjust when you have data)
        # img_path = self.image_paths[idx]
        # mask_path = self.mask_paths[idx]
        # image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # Ensure 4 channels
        # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # DUMMY DATA FOR TESTING
        image = np.random.rand(256, 256, 4).astype(np.float32)
        mask = np.random.randint(0, 24, (256, 256)).astype(np.uint8)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1) # HWC -> CHW
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()

        return image, mask
```

**Create `train.py`:**
```python
%%writefile train.py
import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from src.dataset import LandCoverDataset
from src.models.variant_b import VariantB, DeepSupervisionLoss

# ... (Paste the full content of src/train_variant_b.py here, 
#      but change the import from src.models.variant_b to src.models.variant_b)
#      Actually, since we made src/models as a package, imports should work if you run from root.

# SIMPLIFIED TRAIN LOOP FOR COLAB
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Init Model
    model = VariantB(in_channels=4, classes=24, deep_supervision=True).to(device)
    criterion = DeepSupervisionLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    
    # Data
    dataset = LandCoverDataset(root_dir="/content/drive/MyDrive/STAFT_Project/data")
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Train
    model.train()
    for epoch in range(5): # Run for 5 epochs as test
        total_loss = 0
        pbar = tqdm(loader)
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})
            
        print(f"Epoch {epoch+1} finished. Avg Loss: {total_loss/len(loader)}")
        
        # Save checkpoint to Drive
        torch.save(model.state_dict(), f"/content/drive/MyDrive/STAFT_Project/model_epoch_{epoch}.pth")

if __name__ == "__main__":
    main()
```

## Step 4: Run Training
```python
!python train.py
```

## Tips
-   **Data Path:** Double check the path in `train.py` matches where you uploaded your data in Drive.
-   **Session Timeout:** Colab sessions disconnect if idle. Keep the browser tab open.
-   **GPU Memory:** If you get "CUDA out of memory", reduce `batch_size` in `train.py` (try 4 or 2).

