import os
import pickle
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import VitPoseForPoseEstimation

# ══════════════════════════════════════════════════════════════════════════════
# Suppress huggingface symlinks warning on Windows
# ══════════════════════════════════════════════════════════════════════════════
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              CONFIGURATION                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class CFG:
    # Paths
    PROJECT = Path(r"D:\Milad_vision\LLDFinder-ml")
    DATA_ROOT = PROJECT / "dataset" / "Fitted_femural_head"
    PICKLE_PATH = PROJECT / "dataset" / "annotations.pkl"

    # Output
    OUTPUT = PROJECT / "runs" / "vitpose"

    # Image settings
    ORIG_SIZE = 640  # Your original image size (640x640)

    # ViTPose model requires exactly 256x192 input (height x width)
    MODEL_IMG_HEIGHT = 256
    MODEL_IMG_WIDTH = 192

    # Model
    MODEL_NAME = "usyd-community/vitpose-plus-base"

    # ViTPose+ Expert Index (MoE - Mixture of Experts)
    # 0: MS COCO, 1: AIC, 2: MPII, 3: AP10K, 4: APT36K, 5: WholeBody
    DATASET_INDEX = 0  # Using COCO expert for general pose

    # Training
    EPOCHS = 100
    BATCH_SIZE = 128
    LR = 1e-4
    PATIENCE = 15
    WEIGHT_DECAY = 0.001

    # Checkpointing
    SAVE_PERIOD = 5
    RESUME = True
    RESUME_FROM = None

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 0
    PIN_MEMORY = torch.cuda.is_available()


# Create output directory
CFG.OUTPUT.mkdir(parents=True, exist_ok=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                                 DATASET                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class KeypointDataset(Dataset):
    def __init__(self, df, split, transform):
        self.df = df[df["type"] == split].reset_index(drop=True)
        self.transform = transform
        self.image_dir = CFG.DATA_ROOT / split / "images"

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.image_dir}")

        print(f"  [{split.upper()}] {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = self.image_dir / row["image_name"]
        image = Image.open(img_path).convert("RGB")

        x = max(0.0, min(1.0, row["x_pos"] / CFG.ORIG_SIZE))
        y = max(0.0, min(1.0, row["y_pos"] / CFG.ORIG_SIZE))

        keypoints = torch.tensor([x, y], dtype=torch.float32)
        image = self.transform(image)

        return image, keypoints


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                               TRANSFORMS                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

train_transform = transforms.Compose([
    transforms.Resize((CFG.MODEL_IMG_HEIGHT, CFG.MODEL_IMG_WIDTH)),
    transforms.RandomHorizontalFlip(0.3),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((CFG.MODEL_IMG_HEIGHT, CFG.MODEL_IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                                  MODEL                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class ViTPoseBaseRegressor(nn.Module):
    def __init__(self, dataset_index=None):
        super().__init__()

        self.dataset_index = dataset_index if dataset_index is not None else CFG.DATASET_INDEX

        print(f"Loading: {CFG.MODEL_NAME}")
        print(f"  Using expert index: {self.dataset_index}")

        self.vitpose = VitPoseForPoseEstimation.from_pretrained(CFG.MODEL_NAME)

        # Freeze all parameters
        for p in self.vitpose.parameters():
            p.requires_grad = False

        # Unfreeze last 2 encoder layers
        for p in self.vitpose.backbone.encoder.layer[-2:].parameters():
            p.requires_grad = True

        hidden = self.vitpose.config.backbone_config.hidden_size

        self.regressor = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    def forward(self, x):
        batch_size = x.shape[0]

        dataset_index = torch.full(
            (batch_size,),
            self.dataset_index,
            dtype=torch.long,
            device=x.device
        )

        out = self.vitpose.backbone(x, dataset_index=dataset_index)
        feature_map = out.feature_maps[-1]  # (B, 192, 768)
        features = feature_map.mean(dim=1)  # (B, 768)

        return self.regressor(features)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                          CHECKPOINT FUNCTIONS                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_loss, patience_counter, path):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_loss": best_loss,
        "patience_counter": patience_counter,
        "dataset_index": model.dataset_index,
    }
    torch.save(checkpoint, path)
    print(f"  Saved: {path.name}")


def load_checkpoint(path, model, optimizer, scheduler, scaler):
    print(f"\nLoading checkpoint: {path.name}")

    checkpoint = torch.load(path, map_location=CFG.DEVICE, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    epoch = checkpoint["epoch"]
    best_loss = checkpoint["best_loss"]
    patience_counter = checkpoint.get("patience_counter", 0)

    if "dataset_index" in checkpoint:
        model.dataset_index = checkpoint["dataset_index"]

    print(f"  Resumed from epoch {epoch}")
    print(f"  Best loss: {best_loss:.6f}")

    return epoch, best_loss, patience_counter


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                            TRAINING FUNCTIONS                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0

    for img, kp in tqdm(loader, desc="Train", leave=False):
        img = img.to(CFG.DEVICE)
        kp = kp.to(CFG.DEVICE)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            preds = model(img)
            loss = criterion(preds, kp)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    total_dist = 0

    with torch.no_grad():
        for img, kp in tqdm(loader, desc="Eval", leave=False):
            img = img.to(CFG.DEVICE)
            kp = kp.to(CFG.DEVICE)

            with torch.amp.autocast('cuda'):
                preds = model(img)
                loss = criterion(preds, kp)

            total_loss += loss.item()

            diff = (preds - kp) * CFG.ORIG_SIZE
            dist = torch.sqrt((diff ** 2).sum(dim=1)).mean()
            total_dist += dist.item()

    return total_loss / len(loader), total_dist / len(loader)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              INFERENCE                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def predict_single(model, image_path):
    """Predict keypoint for a single image."""
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image = val_transform(image).unsqueeze(0).to(CFG.DEVICE)

    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            pred = model(image)

    # Convert to pixel coordinates
    x = pred[0, 0].item() * CFG.ORIG_SIZE
    y = pred[0, 1].item() * CFG.ORIG_SIZE

    return x, y


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                                   MAIN                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main():
    print("=" * 70)
    print("ViTPose-Plus-Base X-Ray Keypoint Detection")
    print("=" * 70)
    print(f"\nDevice: {CFG.DEVICE}")
    print(f"Original image size: {CFG.ORIG_SIZE}x{CFG.ORIG_SIZE}")
    print(f"Model input size: {CFG.MODEL_IMG_HEIGHT}x{CFG.MODEL_IMG_WIDTH}")
    print(f"Expert index: {CFG.DATASET_INDEX}")

    # ─────────────────────────────────────────────────────────────────────────
    # Load Data
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\nLoading data from: {CFG.PICKLE_PATH}")

    with open(CFG.PICKLE_PATH, "rb") as f:
        df = pickle.load(f)

    print(f"  Total: {len(df)} samples")

    train_ds = KeypointDataset(df, "train", train_transform)
    val_ds = KeypointDataset(df, "val", val_transform)
    test_ds = KeypointDataset(df, "test", val_transform)

    train_dl = DataLoader(
        train_ds,
        batch_size=CFG.BATCH_SIZE,
        shuffle=True,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=CFG.PIN_MEMORY
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=CFG.BATCH_SIZE,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=CFG.PIN_MEMORY
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=CFG.BATCH_SIZE,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=CFG.PIN_MEMORY
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Model, Optimizer, Scheduler
    # ─────────────────────────────────────────────────────────────────────────
    print("\nInitializing model...")
    model = ViTPoseBaseRegressor().to(CFG.DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG.LR,
        weight_decay=CFG.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    scaler = torch.amp.GradScaler('cuda')

    # ─────────────────────────────────────────────────────────────────────────
    # Resume from checkpoint
    # ─────────────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_loss = float('inf')
    patience_counter = 0

    if CFG.RESUME:
        if CFG.RESUME_FROM:
            ckpt_path = CFG.OUTPUT / CFG.RESUME_FROM
        else:
            ckpt_path = CFG.OUTPUT / "last.pth"

        if ckpt_path.exists():
            start_epoch, best_loss, patience_counter = load_checkpoint(
                ckpt_path, model, optimizer, scheduler, scaler
            )
            start_epoch += 1

    # ─────────────────────────────────────────────────────────────────────────
    # Training Loop
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)

    for epoch in range(start_epoch, CFG.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{CFG.EPOCHS}")
        print("-" * 40)

        train_loss = train_epoch(model, train_dl, optimizer, criterion, scaler)
        val_loss, val_dist = evaluate(model, val_dl, criterion)

        scheduler.step(val_loss)

        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  Val Dist:   {val_dist:.2f} px")

        # Save last checkpoint
        save_checkpoint(
            model, optimizer, scheduler, scaler,
            epoch, best_loss, patience_counter,
            CFG.OUTPUT / "last.pth"
        )

        # Save periodic checkpoint
        if (epoch + 1) % CFG.SAVE_PERIOD == 0:
            save_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch, best_loss, patience_counter,
                CFG.OUTPUT / f"epoch_{epoch + 1}.pth"
            )

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), CFG.OUTPUT / "best.pth")
            print(f"  New best model saved!")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{CFG.PATIENCE}")

        # Early stopping
        if patience_counter >= CFG.PATIENCE:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    # ─────────────────────────────────────────────────────────────────────────
    # Final Test Evaluation
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Final Test Evaluation")
    print("=" * 70)

    # Load best model
    model.load_state_dict(torch.load(CFG.OUTPUT / "best.pth", map_location=CFG.DEVICE, weights_only=True))

    test_loss, test_dist = evaluate(model, test_dl, criterion)

    print(f"\nTest Loss: {test_loss:.6f}")
    print(f"Test Distance: {test_dist:.2f} px")

    print("\n" + "=" * 70)
    print("Training Complete!")
    print(f"Best model saved to: {CFG.OUTPUT / 'best.pth'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
