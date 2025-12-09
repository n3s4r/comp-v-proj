import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class LandCoverDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            split (string): 'train' or 'val'.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.image_paths = [] # TODO: Populate this list based on actual file structure
        self.mask_paths = []  # TODO: Populate this list based on actual file structure
        
        # Placeholder for loading file lists
        # self._load_files()

    def _load_files(self):
        # Implementation depends on how the dataset is organized on disk
        pass

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Reading 4 channels: Blue, Green, Red, NIR
        # Note: standard cv2.imread loads BGR. NIR usually needs special handling or is in a separate file/channel.
        # Assuming the image is stored as a multi-channel Tiff or similar.
        # For now, we'll assume a placeholder loading mechanism.
        
        # image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) 
        # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Placeholder random data for testing pipeline
        image = np.random.rand(256, 256, 4).astype(np.float32)
        mask = np.random.randint(0, 24, (256, 256)).astype(np.uint8)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Convert to tensor
        # PyTorch expects [C, H, W]
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1)
        
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()

        return image, mask

