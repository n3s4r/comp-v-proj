import os
import shutil
import random

def prepare_small_dataset(source_images_dir, source_masks_dir, dest_dir, num_samples=10):
    """
    Copies a small subset of images and masks to a new directory for training.
    
    Args:
        source_images_dir: Path to 'Image__8bit_NirRGB' folder
        source_masks_dir: Path to 'Annotation__index' folder
        dest_dir: Destination folder (will create 'train' subfolder)
        num_samples: Number of images to use
    """
    
    # Create directories
    train_img_dir = os.path.join(dest_dir, 'train', 'images')
    train_mask_dir = os.path.join(dest_dir, 'train', 'masks')
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_mask_dir, exist_ok=True)
    
    # Get list of images
    # Assuming file names match or have a consistent pattern
    # Based on the drive link, let's assume standard matching names
    all_images = sorted([f for f in os.listdir(source_images_dir) if f.endswith('.tif') or f.endswith('.png')])
    
    if not all_images:
        print("No images found in source directory.")
        return

    # Select random samples
    # If fewer than requested, take all
    selected_images = random.sample(all_images, min(num_samples, len(all_images)))
    
    print(f"Selecting {len(selected_images)} samples...")
    
    for img_name in selected_images:
        # Copy Image
        src_img_path = os.path.join(source_images_dir, img_name)
        dst_img_path = os.path.join(train_img_dir, img_name)
        shutil.copy2(src_img_path, dst_img_path)
        
        # Copy Mask
        # Assumption: Mask has same filename. If not, logic needs adjustment based on 'readme (important).txt'
        mask_name = img_name # Modify if mask naming convention differs
        src_mask_path = os.path.join(source_masks_dir, mask_name)
        dst_mask_path = os.path.join(train_mask_dir, mask_name)
        
        if os.path.exists(src_mask_path):
            shutil.copy2(src_mask_path, dst_mask_path)
        else:
            print(f"Warning: Mask not found for {img_name}")

    print(f"Dataset preparation complete in {dest_dir}")

# Example Usage (Commented out)
# prepare_small_dataset(
#     source_images_dir='/content/drive/MyDrive/Downloads/Image__8bit_NirRGB',
#     source_masks_dir='/content/drive/MyDrive/Downloads/Annotation__index',
#     dest_dir='/content/drive/MyDrive/STAFT_Project/small_data'
# )

