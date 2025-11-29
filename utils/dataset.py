import os
import sys
from pathlib import Path 

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from tqdm.auto import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import glob
import pandas as pd
import numpy as np
from utils.pipeline import Pipeline

DATA_ROOT = "dataset"
CLASSES = ['COVID', 'Healthy', 'Non-COVID']

class ClassificationDataset(Dataset):
    def __init__(self, root, transform, csv_path=None, classes=CLASSES):
        """
        Classification dataset that can load from CSV file or scan directories.
        
        Args:
            root: Root directory of the dataset
            transform: Image transform
            csv_path: Path to CSV file with 'id' and 'class' columns. If None, scans directories.
            classes: List of class names
        """
        self.root = root
        self.classes = classes
        self.transform = transform
        
        if csv_path and os.path.exists(csv_path):
            # Load from CSV file
            df = pd.read_csv(csv_path)
            self.samples = []
            for _, row in df.iterrows():
                image_id = row['id']
                class_name = row['class']
                # Construct full path: root/class/images/id.png
                img_path = os.path.join(root, class_name, "images", f"{image_id}.png")
                if os.path.exists(img_path):
                    label = classes.index(class_name)
                    self.samples.append((img_path, label))
                else:
                    print(f"Warning: Image not found: {img_path}")
        else:
            # Fallback to directory scanning (original behavior)
            self.samples = [(p, i) for i, cls in enumerate(classes)
                                 for p in glob.glob(os.path.join(root, cls, "images", "*.png"))]

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        # Convert PIL to numpy array for Albumentations
        img_np = np.array(img)
        # Apply transform with named argument (Albumentations requirement)
        transformed = self.transform(image=img_np)
        img = transformed['image']
        return img, label

    def __len__(self): return len(self.samples)

class SegmentationDataset(Dataset):
    def __init__(self, root, transform, csv_path=None, classes=CLASSES):
        """
        Segmentation dataset that can load from CSV file or scan directories.
        
        Args:
            root: Root directory of the dataset
            transform: Image and mask transform
            csv_path: Path to CSV file with 'id' and 'class' columns. If None, scans directories.
            classes: List of class names
        """
        self.root = root
        self.classes = classes
        self.transform = transform
        
        if csv_path and os.path.exists(csv_path):
            # Load from CSV file
            df = pd.read_csv(csv_path)
            self.pairs = []
            for _, row in df.iterrows():
                image_id = row['id']
                class_name = row['class']
                # Construct full paths: root/class/images/id.png and root/class/masks/id.png
                img_path = os.path.join(root, class_name, "images", f"{image_id}.png")
                mask_path = os.path.join(root, class_name, "masks", f"{image_id}.png")
                if os.path.exists(img_path) and os.path.exists(mask_path):
                    self.pairs.append((img_path, mask_path))
                else:
                    if not os.path.exists(img_path):
                        print(f"Warning: Image not found: {img_path}")
                    if not os.path.exists(mask_path):
                        print(f"Warning: Mask not found: {mask_path}")
        else:
            # Fallback to directory scanning (original behavior)
            self.pairs = [(os.path.join(root, c, "images", n),
                          os.path.join(root, c, "masks", n))
                        for c in classes for n in os.listdir(os.path.join(root, c, "images"))
                        if n.endswith(".png") and os.path.exists(os.path.join(root, c, "masks", n))]

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        # Convert PIL to numpy arrays for Albumentations
        img_np = np.array(img)
        mask_np = np.array(mask)
        # Apply transform with named arguments (Albumentations requirement)
        transformed = self.transform(image=img_np, mask=mask_np)
        img = transformed['image']
        mask = transformed['mask']
        return img, mask

    def __len__(self): return len(self.pairs)