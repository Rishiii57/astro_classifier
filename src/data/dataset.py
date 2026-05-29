import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image


# Galaxy10 has 10 classes : i merge them into 5 broader galaxy classes
LABEL_MAP = {
    # 0: Disturbed — DROPPED
    1: 0,  # Merging
    2: 1,  # Round Smooth    → Elliptical
    3: 1,  # In-between Smooth → Elliptical
    4: 1,  # Cigar Smooth    → Elliptical
    5: 2,  # Barred Spiral   → Spiral
    6: 2,  # Unbarred Tight  → Spiral
    7: 2,  # Unbarred Loose  → Spiral
    8: 3,  # Edge-on no Bulge → Edge-on
    9: 3,  # Edge-on with Bulge → Edge-on
}

CLASS_NAMES = [
    'Merging',       # 0
    'Elliptical',    # 1
    'Spiral',        # 2
    'Edge-on',       # 3
    'Nebula',        # 4
    'Planetary',     # 5
    'Star Cluster',  # 6
]

KAGGLE_FOLDER_MAP = {
    'nebula - Google Search':  4,  
    'planets - Google Search': 5,  
    'stars - Google Search':   6,  
}

# ImageNet mean/std , required for pretrained EfficientNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(split='train'):
    if split == 'train':
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:  # val and test — no augmentation
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

def get_augmentation_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(360),          
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
    ])


class AstroDataset(Dataset):
    """
    Unified dataset for Galaxy10 (numpy arrays) + Kaggle space images (jpg files).
    All images are stored as uint8 numpy arrays of shape (H, W, 3).
    """
    def __init__(self, images, labels, transform=None):
        self.images = images      
        self.labels = labels      
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]  # uint8 numpy array

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(self.labels[idx], dtype=torch.long)


def load_kaggle_images(kaggle_base_dir, target_count=800):
    """
    Loads images from the 3 Kaggle folders (nebula, planets, stars).
    Augments each class to `target_count` images by repeatedly applying
    random augmentations to the original images.

    Returns:
        images: list of uint8 numpy arrays shape (H, W, 3)
        labels: list of int class indices
    """
    aug_transform = get_augmentation_transform()
    all_images = []
    all_labels = []

    for folder_name, class_idx in KAGGLE_FOLDER_MAP.items():
        folder_path = os.path.join(kaggle_base_dir, folder_name)
        if not os.path.isdir(folder_path):
            print(f"  WARNING: folder not found: {folder_path}")
            continue

        # Load all original images
        originals = []
        for fname in os.listdir(folder_path):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                continue
            try:
                img = Image.open(os.path.join(folder_path, fname)).convert('RGB')
                originals.append(np.array(img, dtype=np.uint8))
            except Exception:
                pass  # skip corrupted files

        n_orig = len(originals)
        print(f"  {folder_name}: {n_orig} original images → augmenting to {target_count}")

        # Add all originals first
        for img in originals:
            all_images.append(img)
            all_labels.append(class_idx)

        # Augment to reach target_count
        needed = target_count - n_orig
        if needed > 0:
            for i in range(needed):
                # Cycle through originals
                src = originals[i % n_orig]
                pil_img = Image.fromarray(src)
                aug_tensor = aug_transform(pil_img)  # (3, 224, 224) float tensor
                # Convert back to uint8 numpy for consistency
                aug_np = (aug_tensor.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                all_images.append(aug_np)
                all_labels.append(class_idx)

    return all_images, all_labels


def load_galaxy10(images_path, labels_path):
    images = np.load(images_path)
    labels = np.load(labels_path)
    
    # Filtering out Disturbed class (original label 0)
    mask = labels != 0
    images = images[mask]
    labels = labels[mask]
    
    # Remaping remaining classes
    labels = np.array([LABEL_MAP[l] for l in labels])
    return images, labels


def get_dataloaders(
    images_path,
    labels_path,
    kaggle_base_dir,
    batch_size=32,
    num_workers=0,
    kaggle_target_count=800,
):
    """
    Builds train/val/test DataLoaders from the combined Galaxy10 + Kaggle dataset.

    Args:
        images_path:        path to galaxy10_images.npy
        labels_path:        path to galaxy10_labels.npy
        kaggle_base_dir:    path to 'data/kaggle_space/space images/'
        batch_size:         batch size for all loaders
        num_workers:        dataloader workers (keep 0 on macOS MPS)
        kaggle_target_count: how many images per Kaggle class after augmentation
    """

    # 1. Load Galaxy10
    print("Loading Galaxy10..")
    g10_images, g10_labels = load_galaxy10(images_path, labels_path)
    print(f"  Galaxy10: {len(g10_images)} images, {len(np.unique(g10_labels))} classes")

    # 2. Load + augment Kaggle images
    print("Loading Kaggle space images...")
    kag_images, kag_labels = load_kaggle_images(kaggle_base_dir, target_count=kaggle_target_count)
    print(f"  Kaggle total: {len(kag_images)} images")

    # 3. Combine into one dataset
    all_images = list(g10_images) + kag_images
    all_labels = list(g10_labels) + kag_labels
    all_labels = np.array(all_labels)

    print(f"\nCombined dataset: {len(all_images)} images, {len(CLASS_NAMES)} classes")
    for i, name in enumerate(CLASS_NAMES):
        count = np.sum(all_labels == i)
        print(f"  [{i}] {name}: {count}")

    indices = np.arange(len(all_images))
    idx_train, idx_temp = train_test_split(
        indices, test_size=0.30, random_state=42, stratify=all_labels
    )
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.50, random_state=42, stratify=all_labels[idx_temp]
    )

    def subset(idx_list):
        imgs = [all_images[i] for i in idx_list]
        lbls = all_labels[idx_list]
        return imgs, lbls

    X_train, y_train = subset(idx_train)
    X_val,   y_val   = subset(idx_val)
    X_test,  y_test  = subset(idx_test)

    print(f"\nSplit → Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # 5. Build datasets
    train_dataset = AstroDataset(X_train, y_train, transform=get_transforms('train'))
    val_dataset   = AstroDataset(X_val,   y_val,   transform=get_transforms('val'))
    test_dataset  = AstroDataset(X_test,  y_test,  transform=get_transforms('test'))

    # 6. WeightedRandomSampler for training — handles class imbalance
    sampler = make_weighted_sampler(y_train)

    # 7. DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=sampler, num_workers=num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=False
    )

    return train_loader, val_loader, test_loader, CLASS_NAMES


# Weighted sampler to handle class imbalance in the combined dataset
def make_weighted_sampler(labels):
    """WeightedRandomSampler so every class is seen equally during training."""
    labels = np.array(labels)
    counts = np.bincount(labels)
    weights_per_class = 1.0 / counts
    sample_weights = weights_per_class[labels]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(sample_weights),
        replacement=True,
    )