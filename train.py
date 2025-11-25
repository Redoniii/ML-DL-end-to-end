import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data_loader import get_dataloaders
from sod_model import SOD_CNN

# Paths to MERGED dataset
merged_image_dir = "C:\Users\beris\OneDrive\Desktop\archive\Merged\Images"
merged_mask_dir = "C:\Users\beris\OneDrive\Desktop\archive\Merged\Masks"

# Hyperparameters
batch_size = 8
lr = 1e-3
num_epochs = 20

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load DataLoaders (train, val, test)
train_loader, val_loader, _ = get_dataloaders(merged_image_dir, merged_mask_dir, batch_size=batch_size)

# Initialize model and optimizer
model = SOD_CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Metrics
def iou(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection + 1e-6
    return (intersection / union).item()

def precision(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    tp = (pred * target).sum()
    pp = pred.sum() + 1e-6
    return (tp / pp).item()

def recall(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    tp = (pred * target).sum()
    ap = target.sum() + 1e-6
    return (tp / ap).item()

def f1_score(p, r):
    return 2 * p * r / (p + r + 1e-6)

# Custom loss: BCE + 0.5*(1 - IoU)
bce_criterion = nn.BCELoss()

def combined_loss(pred, target):
    bce = bce_criterion(pred, target)
    iou_loss = 1 - iou(pred, target)
    return bce + 0.5 * iou_loss

best_val_loss = float('inf')

for epoch in range(1, num_epochs + 1):
    model.train()
    train_loss = 0.0
    train_iou = 0.0
    train_prec = 0.0
    train_rec = 0.0
    train_f1 = 0.0

    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} - Train"):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = combined_loss(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        train_iou += iou(outputs, masks) * images.size(0)
        p = precision(outputs, masks)
        r = recall(outputs, masks)
        train_prec += p * images.size(0)
        train_rec += r * images.size(0)
        train_f1 += f1_score(p, r) * images.size(0)

    train_loss /= len(train_loader.dataset)
    train_iou /= len(train_loader.dataset)
    train_prec /= len(train_loader.dataset)
    train_rec /= len(train_loader.dataset)
    train_f1 /= len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    val_iou = 0.0
    val_prec = 0.0
    val_rec = 0.0
    val_f1 = 0.0

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} - Val"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = combined_loss(outputs, masks)

            val_loss += loss.item() * images.size(0)
            val_iou += iou(outputs, masks) * images.size(0)
            p = precision(outputs, masks)
            r = recall(outputs, masks)
            val_prec += p * images.size(0)
            val_rec += r * images.size(0)
            val_f1 += f1_score(p, r) * images.size(0)

    val_loss /= len(val_loader.dataset)
    val_iou /= len(val_loader.dataset)
    val_prec /= len(val_loader.dataset)
    val_rec /= len(val_loader.dataset)
    val_f1 /= len(val_loader.dataset)

    print(f"Epoch {epoch}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f}, IoU: {train_iou:.4f}, Prec: {train_prec:.4f}, Rec: {train_rec:.4f}, F1: {train_f1:.4f} | "
          f"Val Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_sod_model.pth")
        print("Saved best model!")
