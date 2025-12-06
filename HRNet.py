"""
HRNetW48 Keypoint Detection Training - Minimal Version
"""

import pickle
import random
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATA_ROOT = Path(r"D:\Milad_vision\LLDFinder-ml\dataset\Fitted_femural_head")
ANNOTATIONS_FILE = Path(r"D:\Milad_vision\LLDFinder-ml\dataset\annotations.pkl")
OUTPUT_DIR = Path(r"D:\Milad_vision\LLDFinder-ml\outputs")

EPOCHS = 100
BATCH_SIZE = 4
LR = 1e-4
IMG_SIZE = 640
EARLY_STOP_PATIENCE = 15
SAVE_PERIOD = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


# ==============================================================================
# SEED & DIRS
# ==============================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==============================================================================
# DATASET
# ==============================================================================
class KeypointDataset(Dataset):
    def __init__(self, split='train'):
        # Path: DATA_ROOT / split / images
        self.img_dir = DATA_ROOT / split / "images"

        # Load annotations
        with open(ANNOTATIONS_FILE, 'rb') as f:
            df = pickle.load(f)

        # Filter by split type
        df_split = df[df['type'] == split].reset_index(drop=True)
        self.annotations = df_split.to_dict('records')

        print(f"[{split.upper()}] {len(self.annotations)} samples from {self.img_dir}")

        # Transforms
        base_transforms = [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
        ]
        if split == 'train':
            base_transforms.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))

        base_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.transform = transforms.Compose(base_transforms)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        item = self.annotations[idx]

        img = Image.open(self.img_dir / item['image_name']).convert('RGB')
        img = self.transform(img)

        kp = torch.tensor([item['x_pos'], item['y_pos']], dtype=torch.float32)
        return img, kp


# ==============================================================================
# MODEL
# ==============================================================================
class HRNetW48Regressor(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model('hrnet_w48', pretrained=True, features_only=True)

        # Freeze early stages
        for param in self.backbone.parameters():
            param.requires_grad = False
        for name, param in self.backbone.named_parameters():
            if any(s in name for s in ['stage3', 'stage4', 'transition3', 'final']):
                param.requires_grad = True

        total_ch = sum([f['num_chs'] for f in self.backbone.feature_info])

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(total_ch, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
            nn.Sigmoid()
        )

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"HRNet-W48 loaded | Channels: {total_ch} | Trainable: {trainable:,}")

    def forward(self, x):
        features = self.backbone(x)
        pooled = torch.cat([self.gap(f) for f in features], dim=1)
        return self.head(pooled)


# ==============================================================================
# CHECKPOINT
# ==============================================================================
class CheckpointManager:
    def __init__(self, ckpt_dir):
        self.ckpt_dir = ckpt_dir
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def save(self, state, epoch, best_loss, is_best, is_periodic):
        torch.save(state, self.ckpt_dir / "last.pth")
        if is_best:
            torch.save(state, self.ckpt_dir / "best.pth")
            print(f"  🏆 Best model saved (loss: {best_loss:.6f})")
        if is_periodic:
            torch.save(state, self.ckpt_dir / f"epoch_{epoch}.pth")

    def load(self, model, optimizer, scheduler, scaler):
        ckpt_path = self.ckpt_dir / "last.pth"
        if not ckpt_path.exists():
            print("Starting fresh training.")
            return 0, float('inf'), 0

        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        print(f"Resumed from epoch {ckpt['epoch']}, best_loss: {ckpt['best_loss']:.6f}")
        return ckpt['epoch'] + 1, ckpt['best_loss'], ckpt['patience']


# ==============================================================================
# TRAINING
# ==============================================================================
def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0
    for images, keypoints in tqdm(loader, desc="Train", leave=False):
        images, keypoints = images.to(DEVICE), keypoints.to(DEVICE)
        optimizer.zero_grad()
        with autocast(device_type='cuda'):
            loss = criterion(model(images), keypoints)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def val_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    for images, keypoints in tqdm(loader, desc="Val", leave=False):
        images, keypoints = images.to(DEVICE), keypoints.to(DEVICE)
        with autocast(device_type='cuda'):
            loss = criterion(model(images), keypoints)
        total_loss += loss.item()
    return total_loss / len(loader)


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    set_seed(SEED)

    print("\n=== Loading Data ===")
    train_loader = DataLoader(KeypointDataset('train'), BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(KeypointDataset('val'), BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    print("\n=== Creating Model ===")
    model = HRNetW48Regressor().to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, min_lr=1e-7)
    scaler = GradScaler('cuda')

    ckpt_mgr = CheckpointManager(OUTPUT_DIR / "checkpoints")
    start_epoch, best_loss, patience = ckpt_mgr.load(model, optimizer, scheduler, scaler)

    print("\n=== Training ===")
    for epoch in range(start_epoch, EPOCHS):
        print(f"\nEpoch {epoch}/{EPOCHS - 1}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler)
        val_loss = val_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)

        improved = val_loss < best_loss
        if improved:
            best_loss, patience = val_loss, 0
        else:
            patience += 1

        print(f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | Best: {best_loss:.6f} | Patience: {patience}")

        state = {
            'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
            'best_loss': best_loss, 'patience': patience
        }
        ckpt_mgr.save(state, epoch, best_loss, improved, epoch % SAVE_PERIOD == 0)

        if patience >= EARLY_STOP_PATIENCE:
            print("\n⛔ Early stopping!")
            break

    print("\n🎉 Training complete!")


if __name__ == "__main__":
    main()
