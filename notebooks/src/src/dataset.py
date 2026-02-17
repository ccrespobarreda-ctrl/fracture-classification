# src/dataset.py
from sklearn.model_selection import train_test_split
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_SIZE = 224
BATCH_SIZE = 32
LABEL_COL = "fractured"

train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

class FractureDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.loc[idx, "img_path"]
        label = int(self.df.loc[idx, LABEL_COL])

        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32)

def create_dataloaders(df):
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[LABEL_COL]
    )

    train_ds = FractureDataset(train_df, transform=train_tfms)
    val_ds = FractureDataset(val_df, transform=val_tfms)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader, train_df, val_df
