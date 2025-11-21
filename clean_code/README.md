# LUMENAA: Agentic Multimodal Visual Assistance System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Conference Submission - Clean Code Repository**

---

## 📋 Overview

LUMENAA is a research-grade agentic multimodal AI system designed for visual assistance. The system orchestrates specialized vision, language, and audio components through an intelligent LLM-based controller that dynamically plans and executes tool calls based on natural language queries.

### Key Features

- **🎯 Agentic Architecture**: LLM-driven dynamic planning with tool orchestration
- **👁️ Real-time Computer Vision**: YOLO11-based object detection with 3D spatial mapping (distance, angles)
- **🧠 Vision-Language Model**: Pixtral-12B integration for scene understanding and visual question answering
- **🗣️ Multimodal I/O**: Whisper (STT) and Edge-TTS (TTS) for natural voice interaction
- **👤 Face Recognition**: FaceNet-based multi-person recognition with real-time tracking
- **⏱️ Temporal Memory**: 2-minute rolling window for object tracking and historical queries
- **📊 Experimental Metrics**: Comprehensive performance monitoring and analysis

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   LUMENAA Agent System                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input:  Voice (Whisper STT) + Video Stream (Camera/File)   │
│     ↓                                                        │
│  Controller: LLM-based Planner (Mistral)                     │
│     ↓                                                        │
│  Tools:  Object Detection | Scene Analysis | Face Rec |     │
│          Document OCR | VLM Analysis | Chatbot              │
│     ↓                                                        │
│  Output: Natural Language Response (Edge-TTS)                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

1. **Agent Controller** (`core/controller.py`): LLM-based dynamic planner
2. **Tool Adapters** (`core/adapters/`): Thin wrappers for specialized subsystems
3. **Object Detection** (`core/modules/object_detection/`): YOLO11 + 3D mapping
4. **VLM Integration** (`core/modules/vlm/`): Pixtral-12B for visual understanding
5. **Audio I/O** (`core/modules/audio_input/`, `audio_output/`): STT/TTS pipelines
6. **Face Recognition** (`core/modules/face_recognition/`): FaceNet-based identification
7. **Metrics System** (`core/metrics/`): Performance tracking and analysis

---

## 📁 Directory Structure

```
clean_code/
├── launch.py                      # Main entry point
├── logger_integration.py          # Logging infrastructure
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│
├── core/                          # Core system
│   ├── agent_runner.py           # Agent orchestrator
│   ├── controller.py             # LLM-based controller
│   ├── event_bus.py              # Event-driven communication
│   │
│   ├── adapters/                 # Tool adapters
│   │   ├── object_detector.py
│   │   ├── scene_analysis.py
│   │   ├── pixtral_analysis.py
│   │   ├── document_scan.py
│   │   ├── face_recognition.py
│   │   ├── stt.py
│   │   ├── tts.py
│   │   └── chatbot.py
│   │
│   ├── infrastructure/           # Core infrastructure
│   │   ├── tool_registry.py
│   │   ├── frame_provider.py
│   │   ├── recorder_fsm.py
│   │   ├── device_manager.py
│   │   ├── schemas.py
│   │   ├── unified_logger.py
│   │   └── performance_monitor.py
│   │
│   ├── modules/                  # Backend services
│   │   ├── object_detection/    # YOLO detection + 3D mapping
│   │   ├── vlm/                 # Pixtral/Mistral VLM
│   │   ├── audio_input/         # Whisper STT
│   │   ├── audio_output/        # Edge-TTS
│   │   └── face_recognition/    # FaceNet
│   │
│   └── metrics/                  # Performance tracking
│
├── tests/                        # Evaluation & testing
│   ├── dataset_comparison/      # Multi-dataset detection tests
│   ├── hallucination/           # VLM hallucination testing
│   ├── stt_evaluation/          # STT performance tests
│   ├── tts_evaluation/          # TTS quality tests
│   └── face_recognition/        # Face recognition accuracy
│
├── test_results/                 # Test outputs
│   └── csv_outputs/             # CSV results from all tests
│
├── configs/                      # Configuration files
│   ├── tools.yaml               # Tool definitions
│   └── Objects365.yaml          # Object detection classes
│
├── utils/                        # Utility functions
└── models/                       # Model weights (YOLO)
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster inference)
- Mistral API key (for VLM and LLM features)

### Setup

```bash
# 1. Clone the repository
cd clean_code/

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy language model
python -m spacy download en_core_web_sm

# 5. Set environment variables
export MISTRAL_API_KEY="your_api_key_here"

# 6. Verify YOLO model weights exist
# Ensure models/yolo11n_object365.pt or .onnx is present
```

---

## 📊 Dataset Setup

### For Hallucination Testing

```bash
cd tests/hallucination/
python download_coco.py          # Download COCO validation set
python download_custom.py        # Download custom test images
```

### For Face Recognition Testing

```bash
cd tests/face_recognition/
python download_test_photos.py   # Download standard test dataset
python download_diverse.py       # Download diverse evaluation set
```

### For STT/TTS Testing

Test corpora are generated automatically during evaluation runs.

---

## 🎯 How to Run

### 1. Launch the Main System

```bash
python launch.py
```

**Commands:**
- `r` - Start/stop voice recording
- `q` - Quit the agent
- `Ctrl+C` - Emergency stop

### 2. Example Queries

Once running, you can ask natural language questions:

```
"What objects do you see?"
"Find my keys"
"Where is the person standing?"
"What color is the bottle?"
"Read the text on that sign"
"Who is this person?"
```

### 3. System Logs

All activity is logged to `unified_logs/` with:
- Query processing details
- Detection results
- Performance metrics
- API call statistics

---

## 🧪 Running Tests

### Dataset Comparison (Multi-Dataset Detection)

```bash
cd tests/dataset_comparison/
python official_ultralytics_comparison.py
# Tests on COCO (80 classes), PASCAL VOC (20 classes), LVIS (1203 classes), Open Images v7 (601 classes)
# Results saved to: test_results/csv_outputs/official_ultralytics_comparison.csv
```

### Hallucination Testing (VLM Accuracy)

```bash
cd tests/hallucination/
python test_real_lumenaa.py --ground_truth ground_truth.csv --images test_images/
python generate_visualizations.py --results results/
```

### STT Performance Evaluation

```bash
cd tests/stt_evaluation/
python test_performance.py --queries test_queries.csv
python generate_visualizations.py --results results/
```

### TTS Quality Evaluation

```bash
cd tests/tts_evaluation/
python generate_corpus.py
python generate_audio.py
python evaluate_quality.py
python generate_visualizations.py
```

### Face Recognition Accuracy

```bash
cd tests/face_recognition/
python evaluate_single.py --enrolled enrolled_persons/ --test test_images/
python evaluate_multi.py --enrolled enrolled_persons/ --test test_images/
python generate_figures.py --results results/
```

**Note**: All test results are automatically saved to `test_results/csv_outputs/`

---

## 📈 Reproducibility

### Model Versions

- **Object Detection**: YOLOv11n trained on Objects365
- **VLM**: Pixtral-12B (via Mistral AI API)
- **STT**: OpenAI Whisper Medium
- **TTS**: Edge-TTS (Microsoft)
- **Face Recognition**: FaceNet (MTCNN + InceptionResnetV1)

### Hyperparameters

See `configs/tools.yaml` for all tool configurations.

Key parameters:
- Detection confidence threshold: 0.5
- VLM temperature: 0.3
- Context window: 2 minutes (120 seconds)
- Face recognition threshold: 0.6

### Hardware Specifications

All experiments were conducted on:
- CPU: Intel Core i7 or equivalent
- GPU: NVIDIA RTX 3060 or higher (12GB+ VRAM recommended)
- RAM: 16GB minimum

---

## 📊 Performance Metrics

The system tracks:

- **Query Latency**: End-to-end response time
- **Detection FPS**: Real-time object detection frame rate
- **VLM Call Frequency**: API usage patterns
- **Tracking Optimization**: Cache hit rate for repeated queries
- **Face Recognition Accuracy**: Precision, recall, F1-score
- **STT WER**: Word Error Rate for transcription
- **TTS MOS**: Mean Opinion Score for speech quality

Metrics are automatically logged and can be visualized using the included test scripts.

---

## 🔧 Requirements

Core dependencies:

```
numpy>=1.24.0
opencv-python>=4.9.0.80
torch>=2.1.0
ultralytics>=8.0.0
mistralai>=1.0.0
facenet-pytorch>=2.5.3
openai-whisper>=20231117
edge-tts>=6.1.13
pandas>=2.0.0
spacy>=3.5.0
```

See `requirements.txt` for full list.

---

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{lumenaa2025,
  title={LUMENAA: Agentic Multimodal Visual Assistance System},
  author={[Your Name]},
  booktitle={Proceedings of [Conference Name]},
  year={2025}
}
```

---

## 📄 License

This project is released under the MIT License. See `LICENSE` file for details.

---

## 🙏 Acknowledgments

- YOLO team for object detection framework
- Mistral AI for Pixtral-12B VLM access
- OpenAI for Whisper STT model
- FaceNet team for face recognition architecture
- Microsoft for Edge-TTS

---

## 📧 Contact

For questions or issues, please open an issue in the repository or contact [your-email@domain.com].

---

## 🔄 Version History

- **v1.0.0** (November 2025): Initial conference submission release
  - Core agentic architecture
  - All modules integrated and tested
  - Comprehensive evaluation suite included

---

**Note**: This is the clean, minimal version prepared for conference submission. All unnecessary experimental code, logs, and documentation have been removed to focus on core functionality and reproducibility.
