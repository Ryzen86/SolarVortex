import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
from PIL import Image
import numpy as np

# Import torchmetrics for segmentation evaluation
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassF1Score

# Import your custom CA-FUNet model
from ca_funet import CAFUNet

# ==========================================
# 1. Dataset Template
# ==========================================
class SolarDefectDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Args:
            image_dir: Path to your grayscale EL cell images
            mask_dir: Path to your segmentation masks (where pixel values are class IDs 0-27)
            transform: Optional transforms (augmentations)
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image (assuming grayscale)
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert("L")
        image = np.array(image, dtype=np.float32) / 255.0
        
        # Add channel dimension: [1, H, W]
        image = np.expand_dims(image, axis=0)

        # Load mask
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        mask = Image.open(mask_path)
        mask = np.array(mask, dtype=np.longlong) # Shape: [H, W] (Pixel values = 0 to 27)

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        # Apply transforms if using something like torchvision or albumentations here...
        
        return image, mask

# ==========================================
# 2. Training Loop with Metrics
# ==========================================
def train_model():
    # Hyperparameters
    num_classes = 28
    batch_size = 8
    epochs = 50
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    # TODO: Replace with your actual directory paths!
    train_dataset = SolarDefectDataset(image_dir="path/to/train/images", mask_dir="path/to/train/masks")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 2. Initialize CA-FUNet (Full Config for ablation)
    model = CAFUNet(
        encoder_name='resnet34', 
        in_channels=1, 
        classes=num_classes,
        use_dual_path=True,       # The dual branches
        use_dynamic_weights=True  # The class-aware learned fusion
    ).to(device)

    # 3. Loss, Optimizer, and Metrics
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Initialize metrics
    # JaccardIndex = IoU. F1Score = Dice Score.
    # Note: 'average="macro"' computes the metric per class and then averages them,
    # treating all classes equally regardless of their frequency. This is crucial for imbalanced data!
    iou_metric = MulticlassJaccardIndex(num_classes=num_classes, average="macro").to(device)
    dice_metric = MulticlassF1Score(num_classes=num_classes, average="macro").to(device)

    # 4. Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # Reset metrics at the start of each epoch
        iou_metric.reset()
        dice_metric.reset()
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images) # Shape: [Batch, Classes, Height, Width]
            
            # Calculate loss
            loss = criterion(outputs, masks)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            # Note: outputs are logits. torchmetrics can handle logits directly!
            iou_metric.update(outputs, masks)
            dice_metric.update(outputs, masks)
            
            running_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
        # Compute epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_iou = iou_metric.compute().item()
        epoch_dice = dice_metric.compute().item()
        
        print(f"--- Epoch {epoch+1} Summary ---")
        print(f"Average Loss: {epoch_loss:.4f}")
        print(f"Macro mIoU:   {epoch_iou:.4f}")
        print(f"Macro Dice:   {epoch_dice:.4f}")
        print(f"---------------------------\n")
        
        # Save model checkpoint
        torch.save(model.state_dict(), f"cafunet_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    # Ensure you have 'timm' and 'torchmetrics' installed:
    # pip install timm torchmetrics
    
    # train_model() # Uncomment this line when you have updated the paths!
    print("Update the dataset paths in the script before running!")
