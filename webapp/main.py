"""
F-YOLO Hybrid PestVision — FastAPI Backend
3-Stage Pipeline: CNN (MobileNetV2) + YOLOv8 + Fuzzy Logic Fusion

All 3 model scores returned independently in every detection response.
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

# ─── Configuration ────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
MODELS_DIR  = BASE_DIR / "models"
STATIC_DIR  = BASE_DIR / "static"

# YOLO model search order
YOLO_MODEL_PATHS = [
    MODELS_DIR / "best_fyolo_hybrid.pt",
    MODELS_DIR / "best.pt",
    PROJECT_DIR / "best.pt",
    PROJECT_DIR / "runs" / "fyolo_hybrid"        / "weights" / "best.pt",
    PROJECT_DIR / "runs" / "fyolo_strict_train"  / "weights" / "best.pt",
    PROJECT_DIR / "runs" / "fyolo_fast_train"    / "weights" / "best.pt",
    PROJECT_DIR / "runs" / "fyolo_balanced_train" / "weights" / "best.pt",
]

# CNN model search order
CNN_MODEL_PATHS = [
    MODELS_DIR / "cnn_pest_model.h5",
    PROJECT_DIR / "cnn_pest_model.h5",
]

CONFIDENCE_THRESHOLD = 0.25
INFERENCE_SIZE       = 640

# 5 consolidated pest super-categories
CATEGORIES = {
    0: {"name": "Hopper / Cicada",    "key": "Hopper_Cicada",    "color": "#10b981", "icon": "🦗", "severity": "High"},
    1: {"name": "Aphid",              "key": "Aphid",             "color": "#f59e0b", "icon": "🐛", "severity": "Medium"},
    2: {"name": "Borer",              "key": "Borer",             "color": "#ef4444", "icon": "🪲", "severity": "Critical"},
    3: {"name": "Worm / Caterpillar", "key": "Worm_Caterpillar",  "color": "#8b5cf6", "icon": "🐛", "severity": "High"},
    4: {"name": "Beetle / Weevil",    "key": "Beetle_Weevil",     "color": "#06b6d4", "icon": "🪲", "severity": "Medium"},
}

# CNN was trained with ImageDataGenerator (alphabetical order):
#   CNN idx 0 = Aphid, 1 = Beetle_Weevil, 2 = Borer, 3 = Hopper_Cicada, 4 = Worm_Caterpillar
# YOLO CATEGORIES uses a different order (0=Hopper, 1=Aphid, 2=Borer, 3=Worm, 4=Beetle)
# This map converts YOLO class_id → CNN class_id for correct probability lookup
CNN_CLASS_MAP = {
    0: 3,  # YOLO Hopper_Cicada   → CNN index 3 (Hopper_Cicada alphabetically)
    1: 0,  # YOLO Aphid           → CNN index 0
    2: 2,  # YOLO Borer           → CNN index 2
    3: 4,  # YOLO Worm_Caterpillar→ CNN index 4
    4: 1,  # YOLO Beetle_Weevil   → CNN index 1
}
# Reverse map: CNN index → YOLO class_id (for global CNN top-class display)
CNN_TO_YOLO = {v: k for k, v in CNN_CLASS_MAP.items()}

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pestvision")

# ─── Global Model State ───────────────────────────────────────────────────────

yolo_model       = None
cnn_model        = None
fuzzy_sim        = None
yolo_path_used   = None
cnn_path_used    = None
yolo_is_custom   = False
cnn_available    = False
fuzzy_available  = False


# ─── Model Loaders ────────────────────────────────────────────────────────────

def load_yolo():
    global yolo_model, yolo_path_used, yolo_is_custom
    for path in YOLO_MODEL_PATHS:
        if path.exists():
            logger.info(f"✅ YOLO: Loading trained model from: {path}")
            yolo_model     = YOLO(str(path))
            yolo_path_used = str(path)
            yolo_is_custom = True
            return
    logger.warning("⚠️  No trained YOLO weights found. Falling back to yolov8n.pt (COCO pretrained)")
    yolo_model     = YOLO("yolov8n.pt")
    yolo_path_used = "yolov8n.pt (pretrained fallback)"
    yolo_is_custom = False


def load_cnn():
    global cnn_model, cnn_path_used, cnn_available
    for path in CNN_MODEL_PATHS:
        if path.exists():
            try:
                import tensorflow as tf  # noqa
                cnn_model      = tf.keras.models.load_model(str(path))
                cnn_path_used  = str(path)
                cnn_available  = True
                logger.info(f"✅ CNN: Loaded MobileNetV2 from: {path}")
                return
            except Exception as e:
                logger.warning(f"⚠️  CNN load failed: {e}")
    logger.warning("⚠️  CNN model not found — CNN scores will be 'N/A' until you train and place cnn_pest_model.h5")
    cnn_available = False


def load_fuzzy():
    global fuzzy_sim, fuzzy_available
    try:
        import numpy as np
        import skfuzzy as fuzz
        import skfuzzy.control as ctrl

        yolo_conf_var = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'yolo_conf')
        cnn_prob_var  = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'cnn_prob')
        box_area_var  = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'box_area')
        severity_var  = ctrl.Consequent(np.arange(0, 101,  1),    'severity')

        yolo_conf_var['low']    = fuzz.trimf(yolo_conf_var.universe, [0.00, 0.00, 0.40])
        yolo_conf_var['medium'] = fuzz.trimf(yolo_conf_var.universe, [0.25, 0.50, 0.75])
        yolo_conf_var['high']   = fuzz.trimf(yolo_conf_var.universe, [0.60, 1.00, 1.00])

        cnn_prob_var['low']     = fuzz.trimf(cnn_prob_var.universe, [0.00, 0.00, 0.40])
        cnn_prob_var['medium']  = fuzz.trimf(cnn_prob_var.universe, [0.25, 0.50, 0.75])
        cnn_prob_var['high']    = fuzz.trimf(cnn_prob_var.universe, [0.60, 1.00, 1.00])

        box_area_var['small']   = fuzz.trimf(box_area_var.universe, [0.00, 0.00, 0.20])
        box_area_var['medium']  = fuzz.trimf(box_area_var.universe, [0.10, 0.25, 0.50])
        box_area_var['large']   = fuzz.trimf(box_area_var.universe, [0.35, 1.00, 1.00])

        severity_var['low']     = fuzz.trimf(severity_var.universe, [0,   0,  40])
        severity_var['medium']  = fuzz.trimf(severity_var.universe, [25,  50, 75])
        severity_var['high']    = fuzz.trimf(severity_var.universe, [60, 100, 100])

        rules = [
            ctrl.Rule(yolo_conf_var['high']   & cnn_prob_var['high'],    severity_var['high']),
            ctrl.Rule(yolo_conf_var['high']   & cnn_prob_var['medium'],  severity_var['high']),
            ctrl.Rule(yolo_conf_var['medium'] & cnn_prob_var['high'],    severity_var['high']),
            ctrl.Rule(yolo_conf_var['medium'] & cnn_prob_var['medium'],  severity_var['medium']),
            ctrl.Rule(yolo_conf_var['low']    | cnn_prob_var['low'],     severity_var['low']),
            ctrl.Rule(box_area_var['large'],                             severity_var['high']),
            ctrl.Rule(box_area_var['small']   & yolo_conf_var['low'],   severity_var['low']),
            ctrl.Rule(box_area_var['medium']  & yolo_conf_var['medium'],severity_var['medium']),
        ]

        fuzzy_sim      = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))
        fuzzy_available = True
        logger.info("✅ Fuzzy: Logic engine initialized (8 rules)")
    except Exception as e:
        logger.warning(f"⚠️  Fuzzy engine failed to initialize: {e}. Install: pip install scikit-fuzzy")
        fuzzy_available = False


# ─── Inference Helpers ────────────────────────────────────────────────────────

def run_fuzzy(yolo_c: float, cnn_p: float, box_a: float) -> dict:
    """Returns fuzzy severity label and score."""
    if not fuzzy_available:
        score = round(yolo_c * 60 + cnn_p * 30 + box_a * 10, 1)
        label = "Low" if score < 35 else "Medium" if score < 65 else "HIGH"
        return {"label": label, "score": score}
    try:
        fuzzy_sim.input['yolo_conf'] = float(np.clip(yolo_c, 0.001, 0.999))
        fuzzy_sim.input['cnn_prob']  = float(np.clip(cnn_p,  0.001, 0.999))
        fuzzy_sim.input['box_area']  = float(np.clip(box_a,  0.001, 0.999))
        fuzzy_sim.compute()
        score = round(fuzzy_sim.output['severity'], 1)
    except Exception:
        score = round(yolo_c * 60 + cnn_p * 30 + box_a * 10, 1)
    label = "Low" if score < 35 else "Medium" if score < 65 else "HIGH"
    return {"label": label, "score": score}


def run_cnn(image_rgb_np: np.ndarray) -> np.ndarray:
    """
    Returns CNN softmax probabilities array of shape (5,).
    Uses model(arr, training=False) instead of model.predict()
    to avoid TF Metal memory accumulation over long runtimes.
    """
    if not cnn_available or cnn_model is None:
        return np.ones(5) / 5.0
    try:
        import tensorflow as tf  # noqa
        pil_img = Image.fromarray(image_rgb_np).resize((224, 224))
        arr     = np.expand_dims(np.array(pil_img, dtype=np.float32) / 255.0, axis=0)
        # Direct call — no graph accumulation, stable for 1000s of images
        probs   = cnn_model(arr, training=False).numpy()[0]
        return probs
    except Exception as e:
        logger.error(f"CNN inference error: {e}")
        return np.ones(5) / 5.0


def image_to_base64(img_array: np.ndarray, fmt: str = ".jpg") -> str:
    _, buffer = cv2.imencode(fmt, img_array)
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def draw_detections(image: np.ndarray, detections: list) -> np.ndarray:
    annotated = image.copy()
    h, w = annotated.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        color_hex = det["color"]
        r, g, b   = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)
        color_bgr = (b, g, r)
        thickness = max(2, int(min(w, h) / 300))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, thickness)

        # Label includes all 3 model scores
        sev   = det.get("fuzzy_severity", "?")
        label = f"{det['class_name']}  Y:{det['yolo_conf']:.0%}  C:{det['cnn_prob']:.0%}  Sev:{sev}"
        fs    = max(0.45, min(w, h) / 1400)
        ft    = max(1, int(min(w, h) / 700))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)
        ly = max(y1, th + 10)
        cv2.rectangle(annotated, (x1, ly - th - 8), (x1 + tw + 8, ly + 2), color_bgr, -1)
        cv2.putText(annotated, label, (x1 + 4, ly - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), ft)
    return annotated


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_yolo()
    load_cnn()
    load_fuzzy()
    logger.info(f"🚀 PestVision Hybrid API ready")
    logger.info(f"   YOLO: {yolo_path_used}")
    logger.info(f"   CNN : {'loaded' if cnn_available else 'NOT loaded (train + place cnn_pest_model.h5)'}")
    logger.info(f"   Fuzzy: {'ready' if fuzzy_available else 'NOT ready (pip install scikit-fuzzy)'}")
    yield
    logger.info("👋 Shutting down PestVision API")


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="F-YOLO Hybrid PestVision API",
    description="Agricultural pest detection: CNN + YOLOv8 + Fuzzy Logic",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>PestVision</h1><p>Frontend not found.</p>")


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded":      yolo_model is not None,
        "model_path":        yolo_path_used,
        "custom_model":      yolo_is_custom,
        "cnn_loaded":        cnn_available,
        "cnn_path":          cnn_path_used if cnn_available else None,
        "fuzzy_engine":      fuzzy_available,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "inference_size":    INFERENCE_SIZE,
        "categories":        len(CATEGORIES),
        "pipeline":          "CNN + YOLOv8 + Fuzzy",
    }


@app.get("/api/categories")
async def get_categories():
    return {"categories": CATEGORIES}


@app.post("/api/detect")
async def detect_pests(file: UploadFile = File(...)):
    """
    Hybrid pest detection pipeline.
    Returns per-detection: YOLO confidence, CNN probability, Fuzzy severity — all separately.
    """
    if yolo_model is None:
        raise HTTPException(status_code=503, detail="YOLO model not loaded")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")

    try:
        contents = await file.read()
        nparr    = np.frombuffer(contents, np.uint8)
        image    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        h, w = image.shape[:2]
        logger.info(f"📸 {file.filename} ({w}×{h})")

        # ── Stage 1: YOLO Detection ─────────────────────────────────────────
        t0 = time.time()
        results = yolo_model.predict(
            source=image, conf=CONFIDENCE_THRESHOLD,
            imgsz=INFERENCE_SIZE, verbose=False,
            iou=0.4,        # tight NMS — removes overlapping duplicate boxes
            agnostic_nms=True,  # cross-class NMS (same pest won't show twice)
            max_det=20,
        )
        yolo_ms = (time.time() - t0) * 1000

        # ── Stage 2: CNN Global Classification ─────────────────────────────
        t1 = time.time()
        image_rgb  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cnn_probs  = run_cnn(image_rgb)
        cnn_ms     = (time.time() - t1) * 1000
        cnn_top    = int(np.argmax(cnn_probs))
        cnn_top_p  = float(cnn_probs[cnn_top])

        # ── Stage 3: Fuse per YOLO box with Fuzzy ──────────────────────────
        detections = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                yolo_c = float(box.conf[0].item())
                bbox   = box.xyxy[0].tolist()

                # Skip unknown classes (fallback model returns COCO ids)
                if yolo_is_custom and cls_id not in CATEGORIES:
                    continue

                if yolo_is_custom:
                    cat  = CATEGORIES[cls_id]
                    name = cat["name"]
                    color= cat["color"]
                    icon = cat["icon"]
                else:
                    name  = results[0].names.get(cls_id, f"Class {cls_id}")
                    colors= ["#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4"]
                    color = colors[cls_id % len(colors)]
                    icon  = "🔍"

                # CNN prob for this specific class — remap YOLO id → CNN alphabetical id
                cnn_idx = CNN_CLASS_MAP.get(cls_id, cls_id)
                cnn_p   = float(cnn_probs[cnn_idx]) if cnn_idx < len(cnn_probs) else cnn_top_p

                # Box area fraction for fuzzy
                bx1, by1, bx2, by2 = bbox
                area_frac = min(1.0, ((bx2 - bx1) * (by2 - by1)) / (w * h))

                # Fuzzy severity
                fuzz_out = run_fuzzy(yolo_c, cnn_p, area_frac)

                # Combined confidence (60% YOLO, 40% CNN)
                combined = round(0.6 * yolo_c + 0.4 * cnn_p, 4)

                detections.append({
                    "class_id":      cls_id,
                    "class_name":    name,
                    "icon":          icon,
                    "color":         color,
                    "bbox":          [round(v, 1) for v in bbox],
                    # ── 3 Independent Model Scores ─────────────────────────
                    "yolo_conf":     round(yolo_c, 4),
                    "cnn_prob":      round(cnn_p, 4),
                    "combined_conf": combined,
                    # ── Fuzzy Output ───────────────────────────────────────
                    "fuzzy_severity":fuzz_out["label"],
                    "fuzzy_score":   fuzz_out["score"],
                    # Legacy field (used by old card renderer)
                    "confidence":    combined,
                    "severity":      fuzz_out["label"],
                })

        detections.sort(key=lambda d: d["combined_conf"], reverse=True)

        # ── Annotate image ─────────────────────────────────────────────────
        annotated_img = draw_detections(image, detections)

        inference_ms = yolo_ms + cnn_ms
        logger.info(f"✅ {len(detections)} pests — YOLO {yolo_ms:.0f}ms + CNN {cnn_ms:.0f}ms")

        return JSONResponse({
            "success": True,
            "detections":     detections,
            "original_image": image_to_base64(image),
            "annotated_image":image_to_base64(annotated_img),
            "summary": {
                "total_detections":   len(detections),
                "image_size":         [w, h],
                "inference_time_ms":  round(inference_ms, 1),
                "yolo_time_ms":       round(yolo_ms, 1),
                "cnn_time_ms":        round(cnn_ms, 1),
                "model":              yolo_path_used,
                "custom_model":       yolo_is_custom,
                "cnn_loaded":         cnn_available,
                "fuzzy_engine":       fuzzy_available,
                "filename":           file.filename,
                # Global CNN result — CNN top class remapped back to YOLO category name
                "cnn_top_class":      CATEGORIES.get(CNN_TO_YOLO.get(cnn_top, cnn_top), {}).get("name", f"Class {cnn_top}") if yolo_is_custom else f"Class {cnn_top}",
                "cnn_top_confidence": round(cnn_top_p, 4),
            },
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
