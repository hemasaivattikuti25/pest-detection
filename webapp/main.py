"""
F-YOLO PestVision — FastAPI Backend
Agricultural Pest Detection using YOLOv8

No Fuzzy Logic Preprocessing — RAW images only.
"""

import os
import io
import time
import base64
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

# ─── Configuration ───────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"

# Model search order
MODEL_PATHS = [
    MODELS_DIR / "best.pt",
    PROJECT_DIR / "best.pt",
    PROJECT_DIR / "runs" / "fyolo_fast_train" / "weights" / "best.pt",
    PROJECT_DIR / "runs" / "fyolo_balanced_train" / "weights" / "best.pt",
]

CONFIDENCE_THRESHOLD = 0.25
INFERENCE_SIZE = 640

# 5 consolidated pest categories
CATEGORIES = {
    0: {
        "name": "Hopper / Cicada",
        "key": "Hopper_Cicada",
        "color": "#10b981",
        "icon": "🦗",
        "description": "Leafhoppers, planthoppers, cicadas, and lygus bugs. Sap-sucking pests that weaken plants.",
        "severity": "High",
    },
    1: {
        "name": "Aphid",
        "key": "Aphid",
        "color": "#f59e0b",
        "icon": "🐛",
        "description": "Tiny sap-feeding insects that colonize in large numbers. Transmit plant viruses.",
        "severity": "Medium",
    },
    2: {
        "name": "Borer",
        "key": "Borer",
        "color": "#ef4444",
        "icon": "🪲",
        "description": "Larvae that bore into stems and fruits, causing structural damage to crops.",
        "severity": "Critical",
    },
    3: {
        "name": "Worm / Caterpillar",
        "key": "Worm_Caterpillar",
        "color": "#8b5cf6",
        "icon": "🐛",
        "description": "Leaf-feeding larvae including armyworms, cutworms, and grubs. Cause defoliation.",
        "severity": "High",
    },
    4: {
        "name": "Beetle / Weevil",
        "key": "Beetle_Weevil",
        "color": "#06b6d4",
        "icon": "🪲",
        "description": "Hard-shelled insects that damage leaves, roots, and stored grain.",
        "severity": "Medium",
    },
}

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pestvision")

# ─── Model Loading ───────────────────────────────────────────────────────────

model = None
model_path_used = None
model_is_custom = False


def load_model():
    """Load the best available YOLO model."""
    global model, model_path_used, model_is_custom

    for path in MODEL_PATHS:
        if path.exists():
            logger.info(f"✅ Loading trained model from: {path}")
            model = YOLO(str(path))
            model_path_used = str(path)
            model_is_custom = True
            return

    # Fallback to pretrained
    logger.warning("⚠️  No trained model found. Falling back to yolov8n.pt (COCO pretrained)")
    logger.warning("   Place your trained best.pt in: webapp/models/best.pt")
    model = YOLO("yolov8n.pt")
    model_path_used = "yolov8n.pt (pretrained fallback)"
    model_is_custom = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    load_model()
    logger.info(f"🚀 PestVision API ready | Model: {model_path_used}")
    yield
    logger.info("👋 Shutting down PestVision API")


# ─── FastAPI App ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="F-YOLO PestVision API",
    description="Agricultural pest detection powered by YOLOv8",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─── Utility Functions ───────────────────────────────────────────────────────


def image_to_base64(img_array: np.ndarray, fmt: str = ".jpg") -> str:
    """Convert numpy image to base64 data URI."""
    _, buffer = cv2.imencode(fmt, img_array)
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def draw_detections(image: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes and labels on image."""
    annotated = image.copy()
    h, w = annotated.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        color_hex = det["color"]
        # Convert hex to BGR
        r, g, b = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)
        color_bgr = (b, g, r)

        # Draw box
        thickness = max(2, int(min(w, h) / 300))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, thickness)

        # Label background
        label = f"{det['class_name']} {det['confidence']:.0%}"
        font_scale = max(0.5, min(w, h) / 1200)
        font_thickness = max(1, int(min(w, h) / 600))
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

        label_y = max(y1, th + 10)
        cv2.rectangle(annotated, (x1, label_y - th - 10), (x1 + tw + 10, label_y + 4), color_bgr, -1)
        cv2.putText(annotated, label, (x1 + 5, label_y - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    return annotated


# ─── API Endpoints ───────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main SPA page."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>PestVision</h1><p>Frontend not found.</p>")


@app.get("/api/health")
async def health_check():
    """Health check and model status."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": model_path_used,
        "custom_model": model_is_custom,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "inference_size": INFERENCE_SIZE,
        "categories": len(CATEGORIES),
    }


@app.get("/api/categories")
async def get_categories():
    """Return pest category metadata."""
    return {"categories": CATEGORIES}


@app.post("/api/detect")
async def detect_pests(file: UploadFile = File(...)):
    """
    Run pest detection on an uploaded image.
    NO preprocessing — raw image is passed directly to the model.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file (JPEG, PNG, etc.)")

    try:
        # Read and decode image — RAW, no fuzzy logic
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        h, w = image.shape[:2]
        logger.info(f"📸 Processing image: {file.filename} ({w}x{h})")

        # Run inference — RAW image, no preprocessing
        start_time = time.time()
        results = model.predict(
            source=image,
            conf=CONFIDENCE_THRESHOLD,
            imgsz=INFERENCE_SIZE,
            verbose=False,
        )
        inference_ms = (time.time() - start_time) * 1000

        # Parse detections
        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    bbox = box.xyxy[0].tolist()

                    # Map to our categories (for custom model)
                    if model_is_custom and cls_id in CATEGORIES:
                        cat = CATEGORIES[cls_id]
                        detections.append({
                            "class_id": cls_id,
                            "class_name": cat["name"],
                            "key": cat["key"],
                            "confidence": round(conf, 4),
                            "bbox": [round(v, 1) for v in bbox],
                            "color": cat["color"],
                            "icon": cat["icon"],
                            "severity": cat["severity"],
                        })
                    elif not model_is_custom:
                        # Fallback: use COCO class names
                        class_name = result.names.get(cls_id, f"Class {cls_id}")
                        colors = ["#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4"]
                        detections.append({
                            "class_id": cls_id,
                            "class_name": class_name,
                            "key": class_name,
                            "confidence": round(conf, 4),
                            "bbox": [round(v, 1) for v in bbox],
                            "color": colors[cls_id % len(colors)],
                            "icon": "🔍",
                            "severity": "N/A",
                        })

        # Sort by confidence descending
        detections.sort(key=lambda d: d["confidence"], reverse=True)

        # Draw annotations
        annotated_img = draw_detections(image, detections)

        # Convert images to base64
        original_b64 = image_to_base64(image)
        annotated_b64 = image_to_base64(annotated_img)

        logger.info(f"✅ Detected {len(detections)} pests in {inference_ms:.1f}ms")

        return JSONResponse({
            "success": True,
            "detections": detections,
            "original_image": original_b64,
            "annotated_image": annotated_b64,
            "summary": {
                "total_detections": len(detections),
                "image_size": [w, h],
                "inference_time_ms": round(inference_ms, 1),
                "model": model_path_used,
                "custom_model": model_is_custom,
                "filename": file.filename,
            },
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


# ─── Run ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
