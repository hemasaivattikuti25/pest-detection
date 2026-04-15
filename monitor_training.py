#!/usr/bin/env python3
"""
Live training monitor — run this in a separate terminal to watch progress.
Usage: python3 monitor_training.py
"""
import os, time, glob, subprocess

PROJECT = "/Users/sai2005/Downloads/sc_project "
RUNS    = os.path.join(PROJECT, "runs", "fyolo_hybrid_local")
MODELS  = "/Users/sai2005/Downloads/sc_project /webapp/models"
CNN_H5  = os.path.join(MODELS, "cnn_pest_model.h5")
YOLO_PT = os.path.join(MODELS, "best.pt")

def bar(pct, width=25):
    filled = int(width * pct / 100)
    return "█" * filled + "░" * (width - filled)

def check_pid():
    r = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    return "train_local" in r.stdout

def yolo_epoch():
    """Parse latest epoch from YOLO results csv."""
    csv = os.path.join(RUNS, "results.csv")
    if not os.path.exists(csv):
        return None, None, None
    try:
        with open(csv) as f:
            lines = [l.strip() for l in f if l.strip()]
        if len(lines) < 2:
            return None, None, None
        last = lines[-1].split(",")
        epoch = int(float(last[0].strip())) + 1
        # mAP is usually col 6 or 7 depending on ultralytics version
        try:
            map50 = float(last[6].strip()) * 100
        except Exception:
            map50 = 0.0
        return epoch, 80, map50
    except Exception:
        return None, None, None

def cnn_done():
    return os.path.exists(CNN_H5)

def yolo_done():
    return os.path.exists(YOLO_PT)

STAGE = ["label consolidation", "CNN crop build", "CNN training", "YOLO training", "done"]

print("\033[2J\033[H", end="")  # clear screen
print("╔══════════════════════════════════════════════════════╗")
print("║   🌿  F-YOLO Hybrid — Live Training Monitor          ║")
print("║   Press Ctrl+C to stop watching (training continues)  ║")
print("╚══════════════════════════════════════════════════════╝")

try:
    while True:
        print("\033[5;0H", end="")   # move cursor to line 5

        pid_alive = check_pid()
        e, total, map50 = yolo_epoch()
        cnn_ready  = cnn_done()
        yolo_ready = yolo_done()

        print(f"\n  Status : {'🟢 RUNNING' if pid_alive else '🔴 STOPPED'}")
        print(f"  Time   : {time.strftime('%H:%M:%S')}\n")

        # Stage 1
        print(f"  [1/4] Label Consolidation   ✅  Done (15,643 boxes → 5 classes)")

        # Stage 2
        print(f"  [2/4] CNN Crop Dataset      ✅  Done (14,493 train crops)")

        # Stage 3 CNN
        if cnn_ready:
            size = os.path.getsize(CNN_H5) / 1e6
            print(f"  [3/4] CNN (MobileNetV2)     ✅  SAVED  ({size:.1f} MB)")
        else:
            print(f"  [3/4] CNN (MobileNetV2)     🔄  Training...  (saving to webapp/models/)")

        # Stage 4 YOLO
        if yolo_ready:
            size = os.path.getsize(YOLO_PT) / 1e6
            print(f"  [4/4] YOLOv8 Training       ✅  SAVED  ({size:.1f} MB)")
        elif cnn_ready and e:
            pct = min(100, e / total * 100)
            print(f"  [4/4] YOLOv8 Training       🔄  Epoch {e}/{total}  mAP@0.5={map50:.1f}%")
            print(f"        {bar(pct)} {pct:.0f}%")
        elif cnn_ready:
            print(f"  [4/4] YOLOv8 Training       🔄  Starting...")
        else:
            print(f"  [4/4] YOLOv8 Training       ⏳  Waiting for CNN to finish...")

        print()

        if cnn_ready and yolo_ready:
            print("  ✅✅  ALL MODELS TRAINED!")
            print(f"        CNN  → {CNN_H5}")
            print(f"        YOLO → {YOLO_PT}")
            print()
            print("  Run the webapp:")
            print("    cd '/Users/sai2005/Downloads/sc_project /webapp'")
            print("    bash start.sh")
            break

        # Show webapp/models directory listing
        print("  webapp/models/ contents:")
        models_dir = MODELS
        if os.path.exists(models_dir):
            files = os.listdir(models_dir)
            for f in files:
                fp = os.path.join(models_dir, f)
                sz = os.path.getsize(fp) / 1e6
                print(f"    📦 {f:30s}  {sz:.1f} MB")
        else:
            print("    (empty — models will appear here when saved)")

        print("\n" + "─"*55)
        time.sleep(15)

except KeyboardInterrupt:
    print("\n\n  Monitor stopped. Training is still running in background.")
