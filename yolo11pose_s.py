import os
import torch
from pathlib import Path
from ultralytics import YOLO

if __name__ == '__main__':

    # ============== CONFIGURATION ==============
    DATA_PATH = r"D:\Milad_vision\LLDFinder-ml\dataset\Fitted_femural_head\data.yaml"
    PROJECT_DIR = r"D:\Milad_vision\LLDFinder-ml\dataset\Fitted_femural_head\runs\pose"
    EXPERIMENT_NAME = "femoral_head_v1"

    # Checkpoint settings
    SAVE_PERIOD = 5  # Save checkpoint every 5 epochs
    TOTAL_EPOCHS = 100

    # ============== DEVICE SETUP ==============
    if torch.cuda.is_available():
        device = 0
        torch.cuda.empty_cache()  # Clear any residual memory
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    else:
        device = 'cpu'
        print("CUDA not available, using CPU")

    # ============== AUTO-RESUME LOGIC ==============
    # Check if a previous training exists and find last.pt
    experiment_path = Path(PROJECT_DIR) / EXPERIMENT_NAME
    last_checkpoint = experiment_path / "weights" / "last.pt"

    if last_checkpoint.exists():
        print(f"\n{'=' * 50}")
        print(f"📁 Found existing checkpoint: {last_checkpoint}")
        print(f"🔄 RESUMING training from last checkpoint...")
        print(f"{'=' * 50}\n")

        # Load from last checkpoint and resume
        model = YOLO(str(last_checkpoint))

        results = model.train(
            resume=True,  # This is the key - resumes from where it left off
        )
    else:
        print(f"\n{'=' * 50}")
        print(f"🆕 No checkpoint found. Starting FRESH training...")
        print(f"📁 Checkpoints will be saved to: {experiment_path}")
        print(f"{'=' * 50}\n")

        # Load pretrained model and start fresh
        model = YOLO("yolo11s-pose.pt")

        results = model.train(
            data=DATA_PATH,
            epochs=TOTAL_EPOCHS,
            imgsz=512,  # Reduced for 4GB VRAM safety
            batch=2,
            amp=False,  # Disabled for GTX 1650
            patience=15,
            dropout=0.1,
            weight_decay=0.0005,
            augment=True,
            device=device,
            workers=0,  # Windows compatibility
            cache=True,  # Uses your 20GB RAM
            verbose=True,

            # === CHECKPOINT SETTINGS ===
            save=True,  # Enable saving checkpoints
            save_period=SAVE_PERIOD,  # Save every 5 epochs
            project=PROJECT_DIR,  # Where to save runs
            name=EXPERIMENT_NAME,  # Experiment folder name
            exist_ok=True,  # Allow resuming into same folder

            # === ADDITIONAL USEFUL SETTINGS ===
            plots=True,  # Generate training plots
            val=True,  # Validate during training
            cos_lr=True,  # Cosine learning rate scheduler
            close_mosaic=10,  # Disable mosaic last 10 epochs
        )

    # ============== POST-TRAINING ==============
    print("\n" + "=" * 50)
    print("✅ Training Complete!")
    print("=" * 50)

    # Load best model for evaluation
    best_model_path = experiment_path / "weights" / "best.pt"
    if best_model_path.exists():
        best_model = YOLO(str(best_model_path))
        print(f"\n📊 Loaded best model from: {best_model_path}")

        # Validate on validation set
        print("\n🔍 Validating on VAL set...")
        val_results = best_model.val(data=DATA_PATH, split="val")
        print(f"   Val mAP50: {val_results.pose.map50:.4f}")
        print(f"   Val mAP50-95: {val_results.pose.map:.4f}")

        # Test on test set
        print("\n🧪 Testing on TEST set...")
        test_results = best_model.val(data=DATA_PATH, split="test")
        print(f"   Test mAP50: {test_results.pose.map50:.4f}")
        print(f"   Test mAP50-95: {test_results.pose.map:.4f}")

    print("\n" + "=" * 50)
    print("📁 All outputs saved to:")
    print(f"   {experiment_path}")
    print("=" * 50)
