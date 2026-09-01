"""
Local & CLI Blitzball YOLOv8 Training Script (train_model.py)

Fine-tunes YOLOv8 on custom annotated Blitzball datasets.

Usage:
    # 1. Standard training run (epochs=60, imgsz=640, batch=16)
    python train_model.py

    # 2. Custom hyperparameters
    python train_model.py --epochs 80 --batch 32 --imgsz 640 --model yolov8s.pt

    # 3. Force CPU training if no GPU is available
    python train_model.py --device cpu
"""

import argparse
import os
import shutil
import sys
from typing import Optional

try:
    import torch
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


def train_blitzball_detector(
    data_yaml: str = "dataset/data.yaml",
    model_name: str = "yolov8n.pt",
    epochs: int = 60,
    imgsz: int = 640,
    batch: int = 16,
    device: Optional[str] = None,
    run_name: str = "blitzball_detector",
    export_weights: str = "models/blitzball_detector.pt",
) -> None:
    """Train YOLO model and export best weights."""
    if not ULTRALYTICS_AVAILABLE:
        print("[Error] 'ultralytics' or 'torch' is not installed.")
        print("Please run: pip install ultralytics torch")
        sys.exit(1)

    if not os.path.exists(data_yaml):
        print(f"[Error] Dataset config '{data_yaml}' not found.")
        print("Please run 'python prepare_dataset.py' to generate your dataset splits and data.yaml.")
        sys.exit(1)

    # Determine training device
    if device is None:
        if torch.cuda.is_available():
            device_str = "0"
            print(f"[Hardware] CUDA GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            device_str = "cpu"
            print("[Hardware] No CUDA GPU detected. Training on CPU.")
    else:
        device_str = str(device)

    print("\n" + "=" * 65)
    print(f" Starting YOLO Blitzball Training: {model_name}")
    print("=" * 65)
    print(f" - Dataset config: {os.path.abspath(data_yaml)}")
    print(f" - Model backbone: {model_name}")
    print(f" - Epochs:         {epochs}")
    print(f" - Image Size:     {imgsz}x{imgsz}")
    print(f" - Batch Size:     {batch}")
    print(f" - Device:         {device_str}")
    print("=" * 65 + "\n")

    # Load pretrained model
    model = YOLO(model_name)

    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=run_name,
        device=device_str,
        plots=True,
        save=True,
        workers=2 if device_str != "cpu" else 0,
        verbose=True,
    )

    # Evaluate validation metrics
    print("\n" + "=" * 65)
    print(" Evaluating Validation Set...")
    print("=" * 65)
    metrics = model.val()
    print(f"Validation Box mAP50:    {metrics.box.map50:.4f}")
    print(f"Validation Box mAP50-95: {metrics.box.map:.4f}")

    # Export best weights to target destination
    best_weights_path = os.path.join(results.save_dir, "weights", "best.pt")
    if os.path.exists(best_weights_path):
        os.makedirs(os.path.dirname(export_weights), exist_ok=True)
        shutil.copy2(best_weights_path, export_weights)
        print("\n" + "=" * 65)
        print(f" [Success] Exported best model weights to: {os.path.abspath(export_weights)}")
        print("=" * 65)
        print("\nBlitzball Pitch Tracker Pro will now automatically use this custom model!")
    else:
        print(f"[Warning] Could not locate best weights at '{best_weights_path}'.")


def main():
    parser = argparse.ArgumentParser(description="Train custom YOLOv8 Blitzball detector.")
    parser.add_argument("--data", type=str, default="dataset/data.yaml", help="Path to data.yaml.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pretrained YOLO model (default: yolov8n.pt).")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs (default: 60).")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution for training (default: 640).")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16).")
    parser.add_argument("--device", type=str, default=None, help="Device to use (0, cpu, etc.).")
    parser.add_argument("--name", type=str, default="blitzball_detector", help="Experiment name.")
    parser.add_argument("--export", type=str, default="models/blitzball_detector.pt", help="Export target for best.pt.")

    args = parser.parse_args()

    train_blitzball_detector(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        run_name=args.name,
        export_weights=args.export,
    )


if __name__ == "__main__":
    main()
