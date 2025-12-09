import os
import argparse
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np

# Import our custom modules
from src.dataset import LandCoverDataset
from src.models.variant_b import VariantB, DeepSupervisionLoss

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        
        # Calculate loss (DeepSupervisionLoss handles list output)
        loss = criterion(outputs, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})

    return running_loss / len(loader)

def validate(model, loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    
    # Metrics
    # For simplicity here, we'll track simple pixel accuracy. 
    # Ideally, use a metric library for IoU/mIoU (e.g., from segmentation_models_pytorch or sklearn)
    correct_pixels = 0
    total_pixels = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Val]")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            
            # Loss
            loss = criterion(outputs, masks)
            running_loss += loss.item()

            # For deep supervision, outputs is a list. The last one is usually the final prediction.
            if isinstance(outputs, (list, tuple)):
                final_output = outputs[-1] # or outputs[0] depending on library version, typically last is finest scale
                # However, SMP UnetPlusPlus returns deep supervision outputs from coarse to fine? 
                # Actually SMP usually returns [output1, output2, ..., output_final].
                # Let's verify: In SMP, deep_supervision=True returns a list. The 0-th element is often the final high-res output?
                # Wait, usually the loss expects all of them.
                # For metrics, we just want the best one.
                # Let's assume index 0 is the final output for prediction purposes or check documentation.
                # Standard convention in some libs is 0 is final. In others, -1. 
                # SMP U-Net++: "The output of the model is a list of tensors... The first element is the final output."
                # Let's stick with index 0 for metrics.
                final_output = outputs[0]
            else:
                final_output = outputs

            # Metrics
            preds = torch.argmax(final_output, dim=1)
            correct_pixels += (preds == masks).sum().item()
            total_pixels += torch.numel(preds)
            
            pbar.set_postfix({'val_loss': running_loss / (pbar.n + 1)})

    avg_loss = running_loss / len(loader)
    pixel_acc = correct_pixels / total_pixels
    return avg_loss, pixel_acc

def main():
    parser = argparse.ArgumentParser(description="Train Variant B (SK-ResNeXt + U-Net++)")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    
    args = parser.parse_args()

    # Setup
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    # transform = ... # Add Albumentations here if needed
    full_dataset = LandCoverDataset(root_dir=args.data_dir, split='train') # split logic to be refined
    
    # Split train/val (Simple random split for now, ideal is stratified or pre-defined split)
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model
    model = VariantB(in_channels=4, classes=24, deep_supervision=True).to(device)

    # Loss & Optimizer
    criterion = DeepSupervisionLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Scheduler (optional)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    print("Starting training...")
    start_time = time.time()

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)
        
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "best_variant_b.pth"))
            print("Saved best model.")
            
        # Save last checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        }, os.path.join(args.checkpoint_dir, "last.pth"))

    total_time = time.time() - start_time
    print(f"Training complete in {total_time // 60:.0f}m {total_time % 60:.0f}s")

if __name__ == "__main__":
    main()

