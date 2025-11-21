# Dataset Comparison Testing

Official Ultralytics dataset comparison tool for evaluating detection performance across multiple benchmark datasets.

## Overview

This tool downloads and tests object detection performance on:
- **COCO** (80 classes)
- **PASCAL VOC** (20 classes)
- **LVIS** (1203 classes)
- **Open Images v7** (601 classes) - requires FiftyOne

## Usage

### Basic Usage

```bash
cd tests/dataset_comparison
python official_ultralytics_comparison.py
```

You'll be prompted for:
1. **Images per dataset**: Number of images to test from each dataset (default: 75)
2. **Output CSV**: Path to save results (default: `../../test_results/csv_outputs/official_ultralytics_comparison.csv`)

### Requirements

```bash
# Core requirements (already in main requirements.txt)
pip install ultralytics opencv-python pandas tqdm

# Optional: For Open Images v7 support
pip install fiftyone
```

## Features

### Official Ultralytics Methods

- Uses official Ultralytics dataset configurations
- Official download URLs from Ultralytics YAML files
- Compatible with Ultralytics utilities
- Automatic dataset extraction and processing

### Supported Datasets

#### COCO (80 classes)
- Official COCO 2017 validation set
- Annotations from `instances_val2017.json`
- 80 object classes

#### PASCAL VOC (20 classes)
- VOC 2007 and VOC 2012 datasets
- XML annotations
- 20 object classes

#### LVIS (1203 classes)
- LVIS v1 validation set
- 1203 fine-grained categories
- Built on COCO images

#### Open Images v7 (601 classes)
- Requires FiftyOne library
- 601 object classes
- Large-scale dataset

## Output Format

### CSV Columns

- `Dataset`: Dataset name (COCO, PASCAL VOC, LVIS, Open Images v7)
- `Image_ID`: Unique image identifier
- `Image_Path`: Full path to image
- `Ground_Truth_Count`: Total objects in ground truth
- `Detected_Count`: Total objects detected
- `Ground_Truth_Unique_Count`: Unique object classes in ground truth
- `Detected_Unique_Count`: Unique object classes detected
- `Ground_Truth_Objects`: Semicolon-separated list of all ground truth objects
- `Detected_Objects`: Semicolon-separated list of all detected objects
- `Ground_Truth_Unique`: Semicolon-separated list of unique ground truth classes
- `Detected_Unique`: Semicolon-separated list of unique detected classes

## Example Output

```
Dataset         Image_ID  Ground_Truth_Count  Detected_Count  ...
COCO            397133    5                   4               ...
PASCAL VOC 2012 2008_001234  3                2               ...
LVIS            537991    8                   6               ...
Open Images v7  abc123def    12              10              ...
```

## Performance Metrics

The tool automatically calculates:
- Average objects per image (ground truth vs detected)
- Average unique classes per image
- Detection rate (%)
- Per-dataset statistics

## Troubleshooting

### Import Errors

If you see "Could not import ObjectDetectionWithDistanceAngle":
```bash
# Make sure you're in the clean_code directory
cd clean_code/
python tests/dataset_comparison/official_ultralytics_comparison.py
```

### Model Not Found

The script looks for models in `clean_code/models/`:
- `yolo11n_object365.pt` (preferred)
- `yolo11n_object365.onnx` (alternative)

### Open Images v7 Not Available

Install FiftyOne:
```bash
pip install fiftyone
```

### Download Failures

- Check internet connection
- Ensure sufficient disk space (datasets can be large)
- Some downloads may take time (LVIS ~2GB, VOC ~2GB)

## Dataset Storage

Downloaded datasets are stored in:
```
tests/dataset_comparison/datasets/
├── coco/
├── VOC/
├── lvis/
└── open-images-v7/
```

## Advanced Usage

### Custom Model Path

Edit the script to use a different model:
```python
model_path = "path/to/your/model.pt"
comparison = OfficialDatasetComparison(model_path=model_path)
```

### Custom Output Location

```python
# Specify custom CSV path
comparison.generate_csv_report(results, "custom_output.csv")
```

### Programmatic Usage

```python
from official_ultralytics_comparison import OfficialDatasetComparison

# Initialize
comparison = OfficialDatasetComparison(model_path="models/yolo11n_object365.pt")

# Download specific dataset
coco_images = comparison.downloaders['COCO'].download_coco_sample(100)

# Run detection
results = comparison.compare_datasets(coco_images)

# Generate report
comparison.generate_csv_report(results, "coco_only.csv")
```

## Performance Tips

1. **Start Small**: Test with fewer images first (e.g., 25 per dataset)
2. **GPU Recommended**: Detection is faster with CUDA-enabled GPU
3. **Disk Space**: Ensure at least 10GB free for datasets
4. **Memory**: Close other applications if processing large datasets

## Citation

If using these datasets in research:

**COCO**:
```bibtex
@inproceedings{lin2014microsoft,
  title={Microsoft COCO: Common objects in context},
  author={Lin, Tsung-Yi and others},
  booktitle={ECCV},
  year={2014}
}
```

**PASCAL VOC**:
```bibtex
@article{everingham2010pascal,
  title={The PASCAL Visual Object Classes (VOC) challenge},
  author={Everingham, Mark and others},
  journal={IJCV},
  year={2010}
}
```

**LVIS**:
```bibtex
@inproceedings{gupta2019lvis,
  title={LVIS: A dataset for large vocabulary instance segmentation},
  author={Gupta, Agrim and others},
  booktitle={CVPR},
  year={2019}
}
```

**Open Images**:
```bibtex
@article{OpenImages,
  author = {Alina Kuznetsova and others},
  title = {The Open Images Dataset V4},
  journal = {IJCV},
  year = {2020}
}
```
