#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# F-YOLO PestVision — Launch Script
# Usage:  bash start.sh
#         bash start.sh --port 8080
# ─────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=${1:-8000}
HOST="0.0.0.0"

echo ""
echo "  ██████╗ ███████╗███████╗████████╗██╗   ██╗██╗███████╗██╗ ██████╗ ███╗  ██╗"
echo "  ██╔══██╗██╔════╝██╔════╝╚══██╔══╝╚██╗ ██╔╝██║██╔════╝██║██╔═══██╗████╗ ██║"
echo "  ██████╔╝█████╗  ███████╗   ██║    ╚████╔╝ ██║███████╗██║██║   ██║██╔██╗██║"
echo "  ██╔═══╝ ██╔══╝  ╚════██║   ██║     ╚██╔╝  ██║╚════██║██║██║   ██║██║╚████║"
echo "  ██║     ███████╗███████║   ██║      ██║   ██║███████║██║╚██████╔╝██║ ╚███║"
echo "  ╚═╝     ╚══════╝╚══════╝   ╚═╝      ╚═╝   ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚══╝"
echo ""
echo "  🌿 F-YOLO PestVision | Agricultural Pest Detection Engine"
echo "  ─────────────────────────────────────────────────────────"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "  ❌ Python3 not found. Please install Python 3.9+."
    exit 1
fi

PYTHON=$(command -v python3)
echo "  ✅ Python: $($PYTHON --version)"

# Check dependencies
echo "  🔍 Checking dependencies..."
if ! $PYTHON -c "import fastapi, ultralytics, cv2" &>/dev/null; then
    echo "  📦 Installing dependencies..."
    pip install -r "$SCRIPT_DIR/requirements.txt" -q
    echo "  ✅ Dependencies installed"
else
    echo "  ✅ Dependencies OK"
fi

# Check for trained model
MODEL_FOUND=false
MODELS_DIR="$SCRIPT_DIR/models"

for MODEL_FILE in "$MODELS_DIR/best.pt" "$SCRIPT_DIR/../best.pt"; do
    if [ -f "$MODEL_FILE" ]; then
        echo "  ✅ Model found: $MODEL_FILE"
        MODEL_FOUND=true
        break
    fi
done

if [ "$MODEL_FOUND" = false ]; then
    echo ""
    echo "  ⚠️  WARNING: No trained model found in webapp/models/"
    echo "     Falling back to yolov8n.pt (COCO pretrained)"
    echo "     To use your trained weights:"
    echo "       1. Download from Colab: /content/runs/fyolo_fast_train/weights/best.pt"
    echo "       2. Copy to:            webapp/models/best.pt"
    echo ""
fi

# Launch
echo "  🚀 Starting PestVision server..."
echo "  🌐 URL: http://localhost:${PORT}"
echo "  📡 API: http://localhost:${PORT}/api/health"
echo "  ─────────────────────────────────────────────"
echo "  Press Ctrl+C to stop"
echo ""

cd "$SCRIPT_DIR"
$PYTHON -m uvicorn main:app --host "$HOST" --port "${PORT}" --reload
