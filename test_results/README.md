# Test Results

This directory stores all test outputs and results from evaluation scripts.

## Directory Structure

```
test_results/
├── csv_outputs/              # CSV files from all tests
│   ├── official_ultralytics_comparison.csv
│   ├── hallucination_test_results.csv
│   ├── stt_performance_results.csv
│   ├── tts_quality_results.csv
│   └── face_recognition_results.csv
│
├── visualizations/           # Generated plots and figures
├── reports/                  # Generated reports
└── datasets/                 # Downloaded test datasets
```

## CSV Output Files

All evaluation scripts save their results as CSV files in `csv_outputs/`:

### Dataset Comparison
- **official_ultralytics_comparison.csv**: Multi-dataset detection comparison
  - COCO, PASCAL VOC, LVIS, Open Images v7
  - Ground truth vs detected objects

### Scene Analysis Testing
- **scene_analysis_comprehensive.csv**: Complete scene analysis results
  - Scene descriptions and evaluations
  - VQA (Visual Question Answering) results
- **scene_analysis_performance.csv**: Performance metrics summary
  - Latency measurements
  - Throughput analysis
- **scene_analysis_vqa_benchmark.csv**: VQA benchmark results
  - Question-answer pairs
  - Accuracy metrics
- **scene_analysis_cache_effectiveness.csv**: Cache performance
  - Hit rates and effectiveness
  - Memory usage statistics
- **scene_analysis_latency.csv**: Detailed latency profiling
  - Component-level timing
  - Bottleneck analysis
- **scene_analysis_quality.csv**: Scene description quality metrics
  - Semantic similarity scores
  - Quality assessments
- **scene_analysis_statistical.csv**: Statistical analysis
  - Confidence intervals
  - Significance testing

### Hallucination Testing
- **hallucination_test_results.csv**: VLM accuracy metrics
  - Object presence/absence hallucinations
  - Caption accuracy scores

### STT Evaluation
- **stt_performance_results.csv**: Speech-to-text metrics
  - Word Error Rate (WER)
  - Latency measurements
  - Accuracy by query type

### TTS Evaluation
- **tts_quality_results.csv**: Text-to-speech quality
  - Mean Opinion Score (MOS)
  - Naturalness ratings
  - Latency metrics

### Face Recognition
- **face_recognition_results.csv**: Recognition accuracy
  - Precision, Recall, F1-score
  - Single-person and multi-person results

## Usage

All test scripts automatically save their outputs to this directory. No manual configuration needed.

### Running Tests

```bash
# Dataset comparison
cd tests/dataset_comparison
python official_ultralytics_comparison.py

# Scene analysis testing
cd tests/hallucination  # Scene analysis tests are part of VLM testing
python test_pixtral_captions.py  # Generates scene analysis results

# Hallucination testing
cd tests/hallucination
python test_real_lumenaa.py

# STT evaluation
cd tests/stt_evaluation
python test_performance.py

# TTS evaluation
cd tests/tts_evaluation
python evaluate_quality.py

# Face recognition
cd tests/face_recognition
python evaluate_single.py
```

All results will be automatically saved to `test_results/csv_outputs/`.

## Analyzing Results

Use pandas to analyze CSV files:

```python
import pandas as pd

# Load results
df = pd.read_csv('test_results/csv_outputs/official_ultralytics_comparison.csv')

# Analyze by dataset
summary = df.groupby('Dataset').agg({
    'Ground_Truth_Count': 'mean',
    'Detected_Count': 'mean'
})
print(summary)
```

## Notes

- CSV files use UTF-8 encoding
- Timestamp format: `YYYYMMDD_HHMMSS`
- Large files (>100MB) are excluded from git
