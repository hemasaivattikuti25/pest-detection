"""
F-YOLO Hybrid PestVision — LOCAL TRAINING SCRIPT
Optimized for: Apple M3, 8GB RAM, MPS acceleration

Pipeline:
  Stage 1 → CNN  (MobileNetV2, fine-tuned on 5 pest classes)
  Stage 2 → YOLO (YOLOv8n with MPS acceleration)
  Stage 3 → Fuzzy Logic (skfuzzy severity engine)

Run:  python3 train_local.py
"""

import os
import sys
import glob
import shutil
import time
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ─── Paths ────────────────────────────────────────────────────────────────────

# Dataset is already on your machine
PROJECT_DIR  = "/Users/sai2005/Downloads/sc_project "          # trailing space is real
PROJECT_CLEAN = PROJECT_DIR.rstrip()  # no trailing space — for YOLO paths
TRAIN_IMAGES = os.path.join(PROJECT_DIR, "train", "images")
VALID_IMAGES = os.path.join(PROJECT_DIR, "valid", "images")
TRAIN_LABELS = os.path.join(PROJECT_DIR, "train", "labels")
VALID_LABELS = os.path.join(PROJECT_DIR, "valid", "labels")
DATA_YAML    = os.path.join(PROJECT_DIR, "data.yaml")
BALANCED_YAML = os.path.join(PROJECT_DIR, "data_balanced.yaml")

CNN_DATASET  = "/tmp/cnn_pest_dataset"
RUNS_DIR     = os.path.join(PROJECT_DIR, "runs")
WEBAPP_MODELS = "/Users/sai2005/Downloads/sc_project /webapp/models"

CLASS_NAMES  = ["Hopper_Cicada", "Aphid", "Borer", "Worm_Caterpillar", "Beetle_Weevil"]
IMG_SIZE_CNN = 224
BATCH_CNN    = 16        # safe for 8 GB RAM
EPOCHS_CNN   = 15        # enough with fine-tuning
EPOCHS_YOLO  = 80        # M3 MPS: ~30-45 min
BATCH_YOLO   = 8         # safe for 8 GB unified memory

# ─── Step 0: Pre-flight Check ─────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  F-YOLO HYBRID — Local Training (Apple M3 / MPS)")
print("=" * 60)

import torch
device_mps = torch.backends.mps.is_available()
DEVICE = "mps" if device_mps else "cpu"
print(f"  Device  : {DEVICE.upper()}")
print(f"  PyTorch : {torch.__version__}")

# Check dataset
if not os.path.exists(TRAIN_IMAGES):
    print(f"\n❌ Train images not found at: {TRAIN_IMAGES}")
    sys.exit(1)

train_count = len(glob.glob(os.path.join(TRAIN_IMAGES, "*.jpg")) +
                  glob.glob(os.path.join(TRAIN_IMAGES, "*.png")))
val_count   = len(glob.glob(os.path.join(VALID_IMAGES, "*.jpg")) +
                  glob.glob(os.path.join(VALID_IMAGES, "*.png")))
print(f"  Dataset : {train_count} train / {val_count} val images")
print("=" * 60 + "\n")

if train_count == 0:
    print("❌ No training images found. Check your dataset path.")
    sys.exit(1)

# ─── Step 1: Consolidate Labels → 5 Classes ──────────────────────────────────

print("[1/4] Consolidating labels (102 → 5 classes)...")

import yaml

MAPPING_RULES = {
    0: ["hopper", "cicada", "lygus", "leafhopper", "planthopper"],
    1: ["aphid"],
    2: ["borer"],
    3: ["worm", "grub", "spodoptera", "moth", "cutworm", "armyworm", "noctua", "caterpillar"],
    4: ["beetle", "weevil", "cantharis", "chafer", "flea"]
}

# Load original class names
with open(DATA_YAML, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

names = data.get("names", [])
print(f"  Found {len(names)} original classes in data.yaml")

old_to_new = {}
for i, name in enumerate(names):
    name_lower = name.lower()
    for new_id, keywords in MAPPING_RULES.items():
        if any(kw in name_lower for kw in keywords):
            old_to_new[i] = new_id
            break

print(f"  Mapped: {len(old_to_new)} classes → 5 super-classes")

# Rewrite label files (only if not already consolidated)
sample_label = glob.glob(os.path.join(TRAIN_LABELS, "*.txt"))[:1]
already_done = False
if sample_label:
    with open(sample_label[0]) as f:
        first_line = f.readline().strip().split()
    if first_line and int(first_line[0]) < 5:
        # Check if labels look already consolidated
        total_ids = set()
        for lf in glob.glob(os.path.join(TRAIN_LABELS, "*.txt"))[:200]:
            with open(lf) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        total_ids.add(int(parts[0]))
        if max(total_ids) < 5:
            already_done = True
            print("  Labels already consolidated — skipping rewrite.")

if not already_done:
    total_boxes, kept_boxes = 0, 0
    for split_lbl in [TRAIN_LABELS, VALID_LABELS]:
        for txt_file in glob.glob(os.path.join(split_lbl, "*.txt")):
            with open(txt_file, "r") as f:
                lines = f.readlines()
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                total_boxes += 1
                old_id = int(parts[0])
                if old_id in old_to_new:
                    new_lines.append(f"{old_to_new[old_id]} {' '.join(parts[1:])}\n")
                    kept_boxes += 1
            with open(txt_file, "w") as f:
                f.writelines(new_lines)
    print(f"  Boxes kept: {kept_boxes}/{total_boxes}")

# Write balanced YAML
new_data = {
    "path": PROJECT_CLEAN,          # NO trailing space — YOLO strips it
    "train": PROJECT_CLEAN + "/train/images",   # absolute paths
    "val":   PROJECT_CLEAN + "/valid/images",
    "nc":    5,
    "names": CLASS_NAMES
}
with open(BALANCED_YAML, "w") as f:
    yaml.dump(new_data, f, allow_unicode=True, sort_keys=False)
print(f"  ✅ data_balanced.yaml written.\n")


# ─── Step 2: Build CNN Crop Dataset ──────────────────────────────────────────

print("[2/4] Building CNN dataset from bounding box crops...")

from PIL import Image
import numpy as np

CNN_TRAIN = os.path.join(CNN_DATASET, "train")
CNN_VAL   = os.path.join(CNN_DATASET, "val")

# Only rebuild if needed
if os.path.exists(CNN_DATASET):
    existing = sum(
        len(glob.glob(os.path.join(CNN_TRAIN, cls, "*.jpg")))
        for cls in CLASS_NAMES
    )
    if existing > 500:
        print(f"  Reusing existing CNN dataset ({existing} crops).\n")
        skip_cnn_crop = True
    else:
        shutil.rmtree(CNN_DATASET)
        skip_cnn_crop = False
else:
    skip_cnn_crop = False

if not skip_cnn_crop:
    for split, img_dir, lbl_dir, out_dir in [
        ("train", TRAIN_IMAGES, TRAIN_LABELS, CNN_TRAIN),
        ("val",   VALID_IMAGES, VALID_LABELS, CNN_VAL),
    ]:
        for cls in CLASS_NAMES:
            os.makedirs(os.path.join(out_dir, cls), exist_ok=True)

        img_files = (
            glob.glob(os.path.join(img_dir, "*.jpg")) +
            glob.glob(os.path.join(img_dir, "*.png")) +
            glob.glob(os.path.join(img_dir, "*.jpeg"))
        )
        crop_count = 0
        for img_path in img_files:
            base    = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(lbl_dir, base + ".txt")
            if not os.path.exists(lbl_path):
                continue
            try:
                img      = Image.open(img_path).convert("RGB")
                iw, ih   = img.size
                with open(lbl_path) as f:
                    for j, line in enumerate(f):
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        cls_id = int(parts[0])
                        if cls_id >= 5:
                            continue
                        cx, cy, bw, bh = map(float, parts[1:5])
                        x1 = max(0,  int((cx - bw / 2) * iw))
                        y1 = max(0,  int((cy - bh / 2) * ih))
                        x2 = min(iw, int((cx + bw / 2) * iw))
                        y2 = min(ih, int((cy + bh / 2) * ih))
                        if (x2 - x1) < 20 or (y2 - y1) < 20:
                            continue
                        crop = img.crop((x1, y1, x2, y2)).resize((IMG_SIZE_CNN, IMG_SIZE_CNN))
                        save_path = os.path.join(out_dir, CLASS_NAMES[cls_id], f"{base}_{j}.jpg")
                        crop.save(save_path, quality=85)
                        crop_count += 1
            except Exception:
                continue
        print(f"  {split}: {crop_count} crops saved.")

    print("  ✅ CNN crop dataset ready.\n")


# ─── Step 3: Train CNN (MobileNetV2) ─────────────────────────────────────────

CNN_MODEL_PATH = os.path.join(WEBAPP_MODELS, "cnn_pest_model.h5")
os.makedirs(WEBAPP_MODELS, exist_ok=True)

if os.path.exists(CNN_MODEL_PATH):
    print(f"[3/4] CNN model already exists at {CNN_MODEL_PATH}")
    print("  Delete it and re-run to retrain. Skipping CNN training.\n")
else:
    print("[3/4] Training CNN (MobileNetV2 on 5 pest classes)...")
    print(f"  Batch: {BATCH_CNN} | Epochs: {EPOCHS_CNN}")
    print("  Note: CPU training — estimated time ~15-30 min\n")

    try:
        import tensorflow as tf
        print(f"  TensorFlow: {tf.__version__}")

        # Check for Metal GPU
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            print(f"  Metal GPU: {gpus[0].name}")
        else:
            print("  Running on CPU (install tensorflow-metal for GPU)")

        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.layers import (
            GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
        )
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        train_gen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=25,
            width_shift_range=0.15,
            height_shift_range=0.15,
            horizontal_flip=True,
            zoom_range=0.2,
            brightness_range=[0.7, 1.3],
            fill_mode="nearest",
        )
        val_gen = ImageDataGenerator(rescale=1.0 / 255)

        train_flow = train_gen.flow_from_directory(
            CNN_TRAIN,
            target_size=(IMG_SIZE_CNN, IMG_SIZE_CNN),
            batch_size=BATCH_CNN,
            class_mode="categorical",
            shuffle=True,
        )
        val_flow = val_gen.flow_from_directory(
            CNN_VAL,
            target_size=(IMG_SIZE_CNN, IMG_SIZE_CNN),
            batch_size=BATCH_CNN,
            class_mode="categorical",
            shuffle=False,
        )

        print(f"  Train: {train_flow.samples} | Val: {val_flow.samples}")
        print(f"  Classes: {train_flow.class_indices}\n")

        # Build model
        base = MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=(IMG_SIZE_CNN, IMG_SIZE_CNN, 3)
        )
        base.trainable = False

        x   = GlobalAveragePooling2D()(base.output)
        x   = BatchNormalization()(x)
        x   = Dense(256, activation="relu")(x)
        x   = Dropout(0.4)(x)
        x   = Dense(128, activation="relu")(x)
        x   = Dropout(0.3)(x)
        out = Dense(5, activation="softmax")(x)

        cnn = Model(base.input, out)
        cnn.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        callbacks_a = [
            EarlyStopping(patience=4, restore_best_weights=True, monitor="val_accuracy"),
            ReduceLROnPlateau(factor=0.3, patience=2, min_lr=1e-7),
        ]

        # Phase A: head only
        print("  Phase A: Training head (frozen backbone)...")
        cnn.fit(train_flow, validation_data=val_flow,
                epochs=8, callbacks=callbacks_a, verbose=1)

        # Phase B: fine-tune top 30 layers
        print("\n  Phase B: Fine-tuning top 30 layers...")
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False

        cnn.compile(
            optimizer=Adam(learning_rate=5e-5),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        callbacks_b = [
            EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
            ReduceLROnPlateau(factor=0.3, patience=3, min_lr=1e-8),
        ]
        cnn.fit(train_flow, validation_data=val_flow,
                epochs=EPOCHS_CNN, callbacks=callbacks_b, verbose=1)

        val_loss, val_acc = cnn.evaluate(val_flow, verbose=0)
        print(f"\n  ✅ CNN Val Accuracy : {val_acc*100:.2f}%")
        print(f"     CNN Val Loss     : {val_loss:.4f}")

        cnn.save(CNN_MODEL_PATH)
        print(f"  ✅ CNN saved → {CNN_MODEL_PATH}\n")

    except ImportError:
        print("  ⚠️  TensorFlow not installed. Skipping CNN training.")
        print("  Install with: pip3 install tensorflow-macos tensorflow-metal")
        print("  CNN scores will show 'N/A' in webapp until trained.\n")
    except Exception as e:
        print(f"  ❌ CNN training error: {e}\n")


# ─── Step 4: Train YOLOv8 (MPS) ──────────────────────────────────────────────

YOLO_BEST = os.path.join(RUNS_DIR, "fyolo_hybrid_local", "weights", "best.pt")
YOLO_DEST = os.path.join(WEBAPP_MODELS, "best.pt")

if os.path.exists(YOLO_DEST):
    print(f"[4/4] YOLO weights already in webapp/models/best.pt")
    print("  Delete it and re-run to retrain. Skipping YOLO training.\n")
else:
    print("[4/4] Training YOLOv8n (MPS GPU)...")
    print(f"  Device: {DEVICE} | Epochs: {EPOCHS_YOLO} | Batch: {BATCH_YOLO}")
    print("  Estimated time on M3: ~25-45 minutes\n")

    try:
        from ultralytics import YOLO

        yolo = YOLO("yolov8n.pt")
        yolo.train(
            data=BALANCED_YAML,
            epochs=EPOCHS_YOLO,
            imgsz=640,
            batch=BATCH_YOLO,
            device=DEVICE,
            patience=15,
            # Augmentation
            augment=True,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            flipud=0.1,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.05,           # reduced for 8 GB RAM
            copy_paste=0.05,
            # Optimizer
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            weight_decay=0.0005,
            warmup_epochs=3,
            # Output
            project=RUNS_DIR,
            name="fyolo_hybrid_local",
            exist_ok=True,
            save=True,
            plots=True,
            # Memory management for 8 GB
            workers=2,
            cache=False,          # don't cache dataset in RAM to save memory
        )

        # Report metrics
        metrics = yolo.val(data=BALANCED_YAML, imgsz=640, conf=0.25, verbose=False)
        print("\n" + "=" * 50)
        print("  YOLO VALIDATION RESULTS")
        print("=" * 50)
        print(f"  mAP@0.5      : {metrics.box.map50*100:.2f}%")
        print(f"  mAP@0.5:0.95 : {metrics.box.map*100:.2f}%")
        print(f"  Precision    : {metrics.box.mp*100:.2f}%")
        print(f"  Recall       : {metrics.box.mr*100:.2f}%")
        print("=" * 50)

        # Copy to webapp/models/
        if os.path.exists(YOLO_BEST):
            os.makedirs(WEBAPP_MODELS, exist_ok=True)
            shutil.copy(YOLO_BEST, YOLO_DEST)
            print(f"\n  ✅ YOLO weights → {YOLO_DEST}")
        else:
            print(f"  ❌ best.pt not found at {YOLO_BEST}")

    except Exception as e:
        print(f"  ❌ YOLO training error: {e}")
        import traceback; traceback.print_exc()

# ─── Done ─────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  TRAINING COMPLETE")
print("=" * 60)
print(f"  CNN model : {CNN_MODEL_PATH}")
print(f"  YOLO model: {YOLO_DEST}")
print()
print("  Next steps:")
print("  1. cd webapp")
print("  2. pip3 install -r requirements.txt")
print("  3. bash start.sh")
print("  4. Open http://localhost:8000")
print("=" * 60 + "\n")
