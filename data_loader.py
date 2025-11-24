import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import random

class DUTSDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png"))])
        self.masks = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith((".jpg", ".png"))])

        # Match image-mask pairs by filename
        self.data = []
        for img in self.images:
            mask = img.rsplit(".", 1)[0] + ".png"  # mask filename with .png
            if mask in self.masks:
                self.data.append((img, mask))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, mask_name = self.data[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # mask is single channel

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask

# Transforms
class JointTransform:
    def __init__(self, train=True):
        base_transforms = [T.Resize((224,224)), T.ToTensor()]
        if train:
            aug = [T.RandomHorizontalFlip(), T.RandomRotation(10)]
            base_transforms = aug + base_transforms
        self.img_tf = T.Compose(base_transforms)
        self.mask_tf = T.Compose([T.Resize((224,224)), T.ToTensor()])

    def __call__(self, image, mask):
        return self.img_tf(image), self.mask_tf(mask)

def get_transforms(train=True):
    return JointTransform(train=train)

def get_dataloaders(merged_image_dir, merged_mask_dir, batch_size=8):
    """Creates train/val/test DataLoaders from MERGED folder (70/15/15 split)."""
    dataset = DUTSDataset(merged_image_dir, merged_mask_dir, transform=get_transforms(train=True))

    # Split sizes
    total_len = len(dataset)
    train_len = int(0.7 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len

    # Shuffle dataset before split
    indices = list(range(total_len))
    random.shuffle(indices)

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_len, val_len, test_len])

    # Assign transforms
    train_dataset.dataset.transform = get_transforms(train=True)
    val_dataset.dataset.transform = get_transforms(train=False)
    test_dataset.dataset.transform = get_transforms(train=False)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader
