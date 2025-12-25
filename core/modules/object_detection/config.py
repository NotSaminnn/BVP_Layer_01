# Camera Calibration Parameters
# Intrinsic Matrix
CAMERA_FX = 1134.5757  # Focal length in x direction
CAMERA_FY = 1140.2328  # Focal length in y direction
CAMERA_CX = 585.0006   # Principal point x
CAMERA_CY = 469.1131   # Principal point y

# Complete Camera Intrinsic Matrix
CAMERA_MATRIX = [
    [1134.5757,    0.0000,  585.0006],
    [   0.0000, 1140.2328,  469.1131],
    [   0.0000,    0.0000,    1.0000]
]

# Distortion Coefficients
DISTORTION_COEFFICIENTS = [-0.05194337, 0.21849931, -0.03455788, -0.00321696, -0.37679715]

# Calibration Quality
MEAN_REPROJECTION_ERROR = 0.3526

# Model Configuration
MODEL_PATH_ONNX = "yolo11n_object365.onnx"  # ONNX model path (recommended)
MODEL_PATH_PT = "yolo11n_object365.pt"      # PyTorch model path (fallback)
DEFAULT_MODEL_PATH = MODEL_PATH_ONNX        # Default model to use
CONFIG_PATH = "Objects365.yaml"             # YOLO config file

# Detection Parameters
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for detections
REAL_OBJECT_HEIGHT = 1.0    # Assumed real-world height of objects in meters

# Distance Correction Parameters
DISTANCE_CORRECTION_FACTOR = 1  # Camera maps 30cm actual to 3m detected (3m/30cm = 10, so correction = 1/10 = 0.1)

# System Requirements
REQUIRED_OPENCV_VERSION = "4.8.0"  # Minimum OpenCV version required
REQUIRED_PYTHON_VERSION = "3.8"    # Minimum Python version required
GUI_SUPPORT_REQUIRED = True         # Whether GUI support is required for video display

# Video Parameters
DEFAULT_VIDEO_SOURCE = 0    # 0 for webcam, or path to video file
VIDEO_WIDTH = 1280          # Video width in pixels
VIDEO_HEIGHT = 720          # Video height in pixels
SHOW_FPS = True             # Display FPS counter

# Display Parameters
BOUNDING_BOX_COLOR = (0, 255, 0)  # Green color for bounding boxes
TEXT_COLOR = (0, 255, 0)          # Green color for text
TEXT_SCALE = 0.6                  # Text scale
TEXT_THICKNESS = 2                # Text thickness

# Error Handling and Validation
ENABLE_SYSTEM_VALIDATION = True    # Enable system requirements validation
ENABLE_MODEL_VALIDATION = True     # Enable model file validation
ENABLE_CAMERA_VALIDATION = True    # Enable camera access validation
VERBOSE_INITIALIZATION = True      # Show detailed initialization information

# Performance Settings
ENABLE_GPU_ACCELERATION = True     # Enable GPU acceleration if available
MAX_FRAME_SKIP = 2                 # Maximum frames to skip for performance
MEMORY_OPTIMIZATION = True         # Enable memory optimization features
