# 🚀 Quick Start Guide

This guide will help you get LUMENAA running in under 5 minutes.

---

## Prerequisites Checklist

- [ ] Python 3.8+ installed
- [ ] Mistral API key available
- [ ] 8GB+ RAM
- [ ] Webcam (for live demo) or video file

---

## Step 1: Install Dependencies (2 minutes)

```bash
# Navigate to clean_code directory
cd clean_code/

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm
```

---

## Step 2: Set API Key (30 seconds)

### Windows (PowerShell):
```powershell
$env:MISTRAL_API_KEY="your_api_key_here"
```

### Linux/Mac:
```bash
export MISTRAL_API_KEY="your_api_key_here"
```

### Or create `.env` file:
```bash
echo "MISTRAL_API_KEY=your_api_key_here" > .env
```

---

## Step 3: Run the System (30 seconds)

```bash
python launch.py
```

**Expected output**:
```
============================================================
LUMENAA Agent with Unified Logging & Experimental Metrics
============================================================
✓ Experimental metrics system initialized
✓ Unified logging initialized
✓ Session ID: 20251121_143022_a7b3
✓ Log file: unified_logs/session_20251121_143022_a7b3.json
✓ System monitoring started
✓ Agent initialization complete

Commands:
  'r' - Start/stop recording
  'q' - Quit agent
  'Ctrl+C' - Emergency stop
...
Starting agent...
============================================================
```

---

## Step 4: Test Voice Interaction (1 minute)

1. Press **`r`** to start recording
2. Speak your query (e.g., "What objects do you see?")
3. Press **`r`** again to stop recording
4. Wait for response

### Example Queries:
- "What do you see?"
- "Find my keys"
- "Where is the person?"
- "What color is the bottle on the left?"
- "Read the text on that sign"

---

## Troubleshooting

### Import Errors

If you see import errors:
```bash
python fix_imports.py
```

### Missing Model Weights

Ensure YOLO weights exist:
```bash
ls models/yolo11n_object365.pt
# or
ls models/yolo11n_object365.onnx
```

If missing, download from the original project.

### API Key Not Set

Verify your API key:
```bash
# Windows
echo $env:MISTRAL_API_KEY

# Linux/Mac
echo $MISTRAL_API_KEY
```

### Module Not Found

Ensure you're in the `clean_code/` directory and virtual environment is activated:
```bash
cd clean_code/
source venv/bin/activate  # Windows: venv\Scripts\activate
```

---

## Testing the System

### Quick Test (No API calls needed)

Test object detection only:
```python
from core.modules.object_detection.main import ObjectDetectionWithDistanceAngle
import cv2

detector = ObjectDetectionWithDistanceAngle()
detector.initialize()

# Test with webcam
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
detections = detector.detect(frame)
print(f"Detected {len(detections)} objects")
```

### Run Evaluation Tests

```bash
# Test hallucination detection
cd tests/hallucination/
python test_real_lumenaa.py

# Test STT performance
cd tests/stt_evaluation/
python test_performance.py

# Test face recognition
cd tests/face_recognition/
python evaluate_single.py
```

---

## System Requirements

### Minimum:
- CPU: Intel i5 or equivalent
- RAM: 8GB
- GPU: Not required (CPU mode available)
- Storage: 5GB

### Recommended:
- CPU: Intel i7 or equivalent
- RAM: 16GB
- GPU: NVIDIA RTX 3060 (12GB VRAM)
- Storage: 10GB

---

## What's Next?

✅ **System Running?** Try the example queries above  
✅ **Want to Test?** Run the evaluation scripts in `tests/`  
✅ **Need Details?** Read the full `README.md`  
✅ **Found Issues?** Check `EXTRACTION_SUMMARY.md` for troubleshooting  

---

## Quick Reference

| Command | Action |
|---------|--------|
| `python launch.py` | Start the agent |
| Press `r` | Toggle voice recording |
| Press `q` | Quit agent |
| `Ctrl+C` | Emergency stop |
| `python fix_imports.py` | Fix import paths |

---

## Performance Tips

1. **Use GPU**: Install PyTorch with CUDA for faster detection
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Reduce Detection FPS**: Edit detection loop sleep interval if CPU is overloaded

3. **Use ONNX Model**: For faster inference, use `.onnx` model instead of `.pt`

4. **Disable Visual Monitor**: Set `AGENT_VISUAL_MONITOR=0` to disable live display

---

## Support

- 📖 Full documentation: `README.md`
- 🔧 Troubleshooting: `EXTRACTION_SUMMARY.md`
- 🧪 Testing guide: `tests/` directories
- 📊 Architecture: See README architecture section

---

**Time to first response**: ~3-5 minutes from installation to running system! 🎉
