import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data_loader import get_dataloaders
from sod_model import SOD_CNN

# Paths to MERGED dataset
merged_image_dir = "C:/Users/beris/OneDrive/Desktop/archive/Merged/Images"
merged_mask_dir = "C:/Users/beris/OneDrive/Desktop/archive/Merged/Masks"

# Hyperparameters
batch_size = 8
lr = 1e-3
num_epochs = 20

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load DataLoaders (train, val, test)
train_loader, val_loader, _ = get_dataloaders(merged_image_dir, merged_mask_dir, batch_size=batch_size)

# Initialize model, loss, optimizer
model = SOD_CNN().to(device)
criterion = nn.BCELoss()  # Binary Cross-Entropy
optimizer = optim.Adam(model.parameters(), lr=lr)

# IoU metric
def iou(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection + 1e-6
    return (intersection / union).item()

best_val_loss = float('inf')

for epoch in range(1, num_epochs+1):
    model.train()
    train_loss = 0.0
    train_iou = 0.0
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} - Train"):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        train_iou += iou(outputs, masks) * images.size(0)

    train_loss /= len(train_loader.dataset)
    train_iou /= len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    val_iou = 0.0
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} - Val"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            val_loss += loss.item() * images.size(0)
            val_iou += iou(outputs, masks) * images.size(0)

    val_loss /= len(val_loader.dataset)
    val_iou /= len(val_loader.dataset)

    print(f"Epoch {epoch}/{num_epochs} | Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f} | Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_sod_model.pth")
        print("Saved best model!")
