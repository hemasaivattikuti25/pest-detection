<div align="center">
  <img src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png" alt="YOLOv8 Banner" width="100%">
  
  # 🌿 F-YOLO PestVision
  
  **An applied AI engineering pipeline for high-precision agricultural pest detection utilizing YOLOv8n, deployed as a full-stack, real-time web application.**
  
  [![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://python.org)
  [![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF9D00.svg?logo=ultralytics&logoColor=black)](https://ultralytics.com)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.104.0%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
  [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
</div>

<br>

## 📖 Overview

Agricultural pest detection suffers from severe class imbalance and extreme natural camouflage. **F-YOLO PestVision** addresses these computer vision challenges by engineering a consolidated, statistically viable dataset and deploying an optimized YOLOv8 Nano architecture. 

The resulting model is packaged into a highly responsive, glassmorphism-themed frontend backed by a robust asynchronous FastAPI architecture—enabling inference on raw field crop images in milliseconds.

---

## 🔬 Data Engineering & Pipeline Strategy

The original raw dataset consisted of ~19,000 images representing 102 highly imbalanced, fragmented insect categories. Attempting to train a 102-class model resulted in mediocre precision due to the long-tail distribution of rare pests.

### 1. Class Consolidation Strategy
Through aggressive mapping and label translation, the 102 fragmented classes were consolidated into **5 dense, high-frequency super-categories**, ensuring deep statistical viability and dramatic improvements in `mAP@0.5`:

| ID | Super-Category | Keywords / Species Included | Severity |
|:---:|:---|:---|:---:|
| `0` | **Hopper_Cicada** | Leafhoppers, Planthoppers, Cicadas, Lygus bugs | 🔴 High |
| `1` | **Aphid** | Aphids | 🟡 Mid |
| `2` | **Borer** | Borers (stem, fruit) | 🚨 Crit |
| `3` | **Worm_Caterpillar**| Worms, Grubs, Spodoptera, Moths, Cutworms | 🔴 High |
| `4` | **Beetle_Weevil** | Beetles, Weevils, Cantharis, Chafer, Flea | 🟡 Mid |

*(Note: 47 statistically insignificant classes were dropped entirely to prevent model confusion.)*

### 2. Empirical Lessons: Bypassing Fuzzy Preprocessing
Early trials experimented with Zadeh’s Fuzzy Intensification Operator to enhance image contrast. Empirical testing proved that mathematical static thresholds destroyed the natural ridges, shadows, and edge textures of bright pests (like white grubs), resulting in flat graphical blobs.

> **Crucial Architecture Decision:** F-YOLO PestVision relies strictly on **RAW image pixels**. We trust the Deep Convolutional Neural Network (CNN) to extract its own high-dimensional feature maps natively without destructive mathematical interference.

### 3. YOLOv8 Training Optimization
* **Model:** YOLOv8 Nano (`yolov8n.pt`)
* **Resolution:** 640px (downscaling to 320px previously caused catastrophic feature starvation)
* **Epochs:** 100
* **Patience:** 20 (Early stopping)
* **Backbone:** Unfrozen (locking base feature extractors degraded confidence to 15-20%)

---

## 💻 Full-Stack Web Application

The F-YOLO pipeline is paired with a production-grade inference server.

### Tech Stack
* **Backend:** FastAPI (Python), Uvicorn, OpenCV headless, Ultralytics
* **Frontend:** Vanilla JavaScript, HTML5 Canvas, modern CSS3 (Glassmorphism layout, dynamic animations)
* **Design System:** Custom Dark UI with Emerald Green (`#10b981`) agricultural accents, animated confidence scales, and drag-and-drop file ingestion.

### API Architecture
* `/api/detect` `[POST]`: Accepts a raw image, runs YOLOv8 batched inference at a custom 0.25 confidence threshold, and returns bounding boxes, normalized coordinates, severity levels, and base64 encoded annotated images asynchronously.
* `/api/health` `[GET]`: Monitors model initialization, cold-start states, and fallback status.

---

## 🚀 Installation & Local Deployment

### 1. Clone the Repository
```bash
git clone https://github.com/hemasaivattikuti25/pest-detection.git
cd pest-detection/webapp
```

### 2. Install Dependencies
Ensure you have Python 3.9+ installed natively.
```bash
pip install -r requirements.txt
```

### 3. Load Trained Weights
Due to size limits, `.pt` weights are excluded from Git. Ensure you move your heavily-trained Colab weights here:
```bash
mv /path/to/your/trained/best.pt ./models/best.pt
```
*(If `best.pt` is missing, the backend will gracefully fallback to the pretrained COCO `yolov8n.pt` for general bounding box UI testing).*

### 4. Ignite the FastAPI Server
```bash
bash start.sh
# Or manually: uvicorn main:app --host 0.0.0.0 --port 8000
```
Visit `http://localhost:8000` in your browser.

---

## 📜 Repository Structure
```text
.
├── .gitignore
├── Untitled0.ipynb                 # Original Data Science & Training Notebook
├── data.yaml                       # YOLO dataset config (102 classes)
└── webapp/
    ├── main.py                     # FastAPI Async Backend Server
    ├── requirements.txt            # Dependency graph
    ├── start.sh                    # Deployment initializer
    ├── models/
    │   └── best.pt                 # <--- DROP CUSTOM WEIGHTS HERE
    └── static/
        ├── index.html              # F-YOLO Single Page App (SPA)
        ├── css/style.css           # Premium dark-theme variables & glassmorphism
        └── js/app.js               # Async fetch logic & Canvas rendering
```

---

<div align="center">
  <b>Designed for modern agricultural innovation. 🚜</b>
</div>
