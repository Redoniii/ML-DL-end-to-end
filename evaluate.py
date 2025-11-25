import torch
from torch.utils.data import DataLoader
from data_loader import get_dataloaders
from sod_model import SOD_CNN
import matplotlib.pyplot as plt
import os

# Paths to MERGED dataset
merged_image_dir = "C:/Users/beris/OneDrive/Desktop/archive/Merged/Images"
merged_mask_dir  = "C:/Users/beris/OneDrive/Desktop/archive/Merged/Masks"

# Hyperparameters
batch_size = 8

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load DataLoaders (train, val, test)
_, _, test_loader = get_dataloaders(merged_image_dir, merged_mask_dir, batch_size=batch_size)

# Load model
model = SOD_CNN().to(device)
model.load_state_dict(torch.load("best_sod_model.pth", map_location=device))
model.eval()

# Metrics functions
def iou(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection + 1e-6
    return (intersection / union).item()

def precision(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    true_positive = (pred * target).sum()
    predicted_positive = pred.sum() + 1e-6
    return (true_positive / predicted_positive).item()

def recall(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    true_positive = (pred * target).sum()
    actual_positive = target.sum() + 1e-6
    return (true_positive / actual_positive).item()

def f1_score(p, r):
    return 2 * p * r / (p + r + 1e-6)

def mean_absolute_error(pred, target):
    return torch.abs(pred - target).mean().item()

# Create folder to save visualizations
vis_dir = "C:/Users/beris/OneDrive/Desktop/archive/Merged/visualizations"
os.makedirs(vis_dir, exist_ok=True)

# Evaluation loop
all_metrics = []
with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        for i in range(images.size(0)):
            out = outputs[i]
            m = masks[i]

            iou_val = iou(out, m)
            prec = precision(out, m)
            rec = recall(out, m)
            f1 = f1_score(prec, rec)
            mae = mean_absolute_error(out, m)

            all_metrics.append((iou_val, prec, rec, f1, mae))

# Aggregate metrics
mean_iou = sum([m[0] for m in all_metrics]) / len(all_metrics)
mean_prec = sum([m[1] for m in all_metrics]) / len(all_metrics)
mean_rec = sum([m[2] for m in all_metrics]) / len(all_metrics)
mean_f1 = sum([m[3] for m in all_metrics]) / len(all_metrics)
mean_mae = sum([m[4] for m in all_metrics]) / len(all_metrics)

print(f"Test IoU: {mean_iou:.4f}, Precision: {mean_prec:.4f}, Recall: {mean_rec:.4f}, F1: {mean_f1:.4f}, MAE: {mean_mae:.4f}")

# Visualization of few samples and save figures
num_visualize = 3
for i, (images, masks) in enumerate(test_loader):
    images = images.to(device)
    masks = masks.to(device)
    outputs = model(images)

    for j in range(min(num_visualize, images.size(0))):
        img = images[j].cpu().permute(1,2,0).numpy()
        mask = masks[j].cpu().squeeze().numpy()
        pred = outputs[j].detach().cpu().squeeze().numpy()

        plt.figure(figsize=(10,3))
        plt.subplot(1,4,1)
        plt.imshow(img)
        plt.title("Input Image")
        plt.axis('off')

        plt.subplot(1,4,2)
        plt.imshow(mask, cmap='gray')
        plt.title("Ground Truth")
        plt.axis('off')

        plt.subplot(1,4,3)
        plt.imshow(pred, cmap='gray')
        plt.title("Predicted")
        plt.axis('off')

        plt.subplot(1,4,4)
        plt.imshow(img)
        plt.imshow(pred, cmap='jet', alpha=0.5)
        plt.title("Overlay")
        plt.axis('off')

        # Save figure
        plt.savefig(os.path.join(vis_dir, f"sample_{i}_{j}.png"))
        plt.close()
    break  # Hiq këtë linjë nëse do vizualizosh të gjithë test set-in
