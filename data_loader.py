import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

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
            mask = img.replace(".jpg", ".png")
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
from torchvision import transforms

# JointTransform jashtë funksionit për Windows
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

def get_dataloaders(train_img, train_mask, test_img, test_mask, batch_size=8):
    train_dataset = DUTSDataset(train_img, train_mask, transform=get_transforms(train=True))
    test_dataset = DUTSDataset(test_img, test_mask, transform=get_transforms(train=False))

    # num_workers=0 për Windows
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader
