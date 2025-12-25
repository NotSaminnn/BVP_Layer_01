#!/usr/bin/env python3
"""
Official Ultralytics Dataset Comparison Tool

This version uses the exact official Ultralytics dataset configurations
and download methods for maximum reliability and compatibility.

Supports:
- COCO (80 classes)
- PASCAL VOC (20 classes) 
- LVIS (1203 classes)
- Open Images v7 (601 classes)
"""

import os
import sys
import json
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from tqdm import tqdm
import random
import time
import zipfile
import tarfile

# Import the main detection system
try:
    # Add parent directories to path for clean_code structure
    current_dir = Path(__file__).parent
    clean_code_root = current_dir.parent.parent
    if str(clean_code_root) not in sys.path:
        sys.path.insert(0, str(clean_code_root))
    
    from core.modules.object_detection.main import ObjectDetectionWithDistanceAngle
except ImportError as e:
    print(f"Error: Could not import ObjectDetectionWithDistanceAngle: {e}")
    print("Make sure you're running this from the clean_code directory")
    sys.exit(1)

# Try to import Ultralytics utilities
try:
    from ultralytics.utils.downloads import download
    from ultralytics.utils import ASSETS_URL
    ULTRALYTICS_AVAILABLE = True
    print("✅ Ultralytics utilities available - using official methods")
except ImportError:
    print("⚠️  Ultralytics utilities not available - using fallback methods")
    ULTRALYTICS_AVAILABLE = False

class OfficialDatasetDownloader:
    """Base class using official Ultralytics download methods."""
    
    def __init__(self, base_dir: str = "datasets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
    def download_file_fallback(self, url: str, filepath: Path) -> bool:
        """Fallback download method."""
        try:
            import requests
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, stream=True, headers=headers, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f, tqdm(
                desc=filepath.name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            return True
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return False

class OfficialCOCODownloader(OfficialDatasetDownloader):
    """Official COCO downloader using Ultralytics methods."""
    
    def __init__(self, base_dir: str = "datasets"):
        super().__init__(base_dir)
        self.coco_dir = self.base_dir / "coco"
        self.coco_dir.mkdir(exist_ok=True)
        
    def download_coco_sample(self, num_images: int = 100) -> List[Dict[str, Any]]:
        """Download COCO using official Ultralytics methods."""
        print("🔄 Downloading COCO dataset (official Ultralytics method)...")
        
        try:
            # Official COCO URLs
            annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
            annotations_path = self.coco_dir / "annotations_trainval2017.zip"
            
            if not annotations_path.exists():
                print("Downloading COCO annotations...")
                if ULTRALYTICS_AVAILABLE:
                    download([annotations_url], dir=self.coco_dir, threads=1)
                else:
                    if not self.download_file_fallback(annotations_url, annotations_path):
                        return []
                
                # Extract annotations
                print("Extracting COCO annotations...")
                with zipfile.ZipFile(annotations_path, 'r') as zip_ref:
                    zip_ref.extractall(self.coco_dir)
            
            # Load annotations
            annotations_file = self.coco_dir / "annotations" / "instances_val2017.json"
            if not annotations_file.exists():
                print("❌ COCO annotations not found!")
                return []
                
            print("Loading COCO annotations...")
            with open(annotations_file, 'r') as f:
                coco_data = json.load(f)
            
            # COCO class names (80 classes)
            categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
            print(f"Loaded {len(categories)} COCO categories")
            
            # Group annotations by image
            image_annotations = {}
            for ann in coco_data['annotations']:
                image_id = ann['image_id']
                if image_id not in image_annotations:
                    image_annotations[image_id] = []
                image_annotations[image_id].append(ann)
            
            # Select random images with annotations
            images_with_annotations = [img for img in coco_data['images'] 
                                     if img['id'] in image_annotations]
            selected_images = random.sample(images_with_annotations, 
                                          min(num_images, len(images_with_annotations)))
            
            print(f"Selected {len(selected_images)} COCO images to download")
            
            # Download images
            images_dir = self.coco_dir / "val2017"
            images_dir.mkdir(exist_ok=True)
            
            image_urls = []
            for img_info in selected_images:
                image_path = images_dir / img_info['file_name']
                if not image_path.exists():
                    image_url = f"http://images.cocodataset.org/val2017/{img_info['file_name']}"
                    image_urls.append(image_url)
            
            if image_urls:
                if ULTRALYTICS_AVAILABLE:
                    print("Downloading COCO images (batch)...")
                    download(image_urls, dir=images_dir, threads=4)
                else:
                    print("Downloading COCO images (sequential)...")
                    for url in tqdm(image_urls, desc="Downloading"):
                        filename = Path(url).name
                        self.download_file_fallback(url, images_dir / filename)
            
            # Process downloaded images
            dataset_images = []
            for img_info in selected_images:
                image_path = images_dir / img_info['file_name']
                
                if image_path.exists():
                    # Get ground truth objects
                    ground_truth_objects = []
                    if img_info['id'] in image_annotations:
                        for ann in image_annotations[img_info['id']]:
                            if ann['category_id'] in categories:
                                ground_truth_objects.append(categories[ann['category_id']])
                    
                    if ground_truth_objects:
                        dataset_images.append({
                            'dataset': 'COCO',
                            'image_path': str(image_path),
                            'image_id': img_info['id'],
                            'ground_truth_objects': ground_truth_objects,
                            'annotations': image_annotations.get(img_info['id'], [])
                        })
            
            print(f"✅ Successfully processed {len(dataset_images)} COCO images")
            return dataset_images
            
        except Exception as e:
            print(f"❌ COCO download failed: {e}")
            return []

class OfficialPASCALVOCDownloader(OfficialDatasetDownloader):
    """Official PASCAL VOC downloader using Ultralytics methods."""
    
    def __init__(self, base_dir: str = "datasets"):
        super().__init__(base_dir)
        self.voc_dir = self.base_dir / "VOC"
        self.voc_dir.mkdir(exist_ok=True)
        
    def download_pascal_voc_sample(self, num_images: int = 100) -> List[Dict[str, Any]]:
        """Download PASCAL VOC using official Ultralytics methods."""
        print("🔄 Downloading PASCAL VOC (official Ultralytics method)...")
        
        try:
            # Official VOC URLs from Ultralytics YAML
            if ULTRALYTICS_AVAILABLE:
                urls = [
                    f"{ASSETS_URL}/VOCtrainval_06-Nov-2007.zip",  # 446MB, 5012 images
                    f"{ASSETS_URL}/VOCtest_06-Nov-2007.zip",      # 438MB, 4953 images
                    f"{ASSETS_URL}/VOCtrainval_11-May-2012.zip",  # 1.95GB, 17126 images
                ]
            else:
                # Fallback URLs
                urls = [
                    "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
                    "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
                ]
            
            images_dir = self.voc_dir / "images"
            images_dir.mkdir(exist_ok=True)
            
            # Check if already downloaded
            voc_path = images_dir / "VOCdevkit"
            if not voc_path.exists():
                print("Downloading PASCAL VOC datasets...")
                if ULTRALYTICS_AVAILABLE:
                    download(urls, dir=images_dir, threads=2, exist_ok=True)
                else:
                    # Download VOC 2012 only for fallback
                    voc_file = images_dir / "VOCtrainval_11-May-2012.tar"
                    if not voc_file.exists():
                        if not self.download_file_fallback(urls[1], voc_file):
                            return []
                    
                    # Extract
                    print("Extracting PASCAL VOC...")
                    with tarfile.open(voc_file, 'r') as tar_ref:
                        tar_ref.extractall(images_dir)
            
            # PASCAL VOC class names (20 classes)
            voc_classes = [
                'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
                'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
                'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
            ]
            
            # Process VOC datasets
            dataset_images = []
            
            for year in ["2012", "2007"]:
                voc_year_path = voc_path / f"VOC{year}"
                if not voc_year_path.exists():
                    continue
                
                print(f"Processing VOC {year}...")
                
                # Get validation images
                val_file = voc_year_path / "ImageSets" / "Main" / "val.txt"
                if not val_file.exists():
                    continue
                
                with open(val_file, 'r') as f:
                    image_ids = [line.strip() for line in f.readlines()]
                
                # Select random subset
                selected_ids = random.sample(image_ids, min(num_images // 2, len(image_ids)))
                
                for image_id in tqdm(selected_ids, desc=f"Processing VOC {year}"):
                    image_path = voc_year_path / "JPEGImages" / f"{image_id}.jpg"
                    annotation_path = voc_year_path / "Annotations" / f"{image_id}.xml"
                    
                    if not image_path.exists() or not annotation_path.exists():
                        continue
                    
                    # Parse XML annotation
                    ground_truth_objects = []
                    try:
                        tree = ET.parse(annotation_path)
                        root = tree.getroot()
                        
                        for obj in root.findall('object'):
                            class_name = obj.find('name').text
                            difficult = obj.find('difficult')
                            # Include only non-difficult objects and valid classes
                            if (class_name and class_name in voc_classes and 
                                (difficult is None or difficult.text != '1')):
                                ground_truth_objects.append(class_name)
                                
                    except Exception as e:
                        continue
                    
                    if ground_truth_objects:
                        dataset_images.append({
                            'dataset': f'PASCAL VOC {year}',
                            'image_path': str(image_path),
                            'image_id': f"{year}_{image_id}",
                            'ground_truth_objects': ground_truth_objects,
                            'annotations': str(annotation_path)
                        })
                
                # Stop if we have enough images
                if len(dataset_images) >= num_images:
                    break
            
            print(f"✅ Successfully processed {len(dataset_images)} PASCAL VOC images")
            return dataset_images
            
        except Exception as e:
            print(f"❌ PASCAL VOC download failed: {e}")
            return []

class OfficialLVISDownloader(OfficialDatasetDownloader):
    """Official LVIS downloader using Ultralytics methods."""
    
    def __init__(self, base_dir: str = "datasets"):
        super().__init__(base_dir)
        self.lvis_dir = self.base_dir / "lvis"
        self.lvis_dir.mkdir(exist_ok=True)
        
    def download_lvis_sample(self, num_images: int = 100) -> List[Dict[str, Any]]:
        """Download LVIS using official Ultralytics methods."""
        print("🔄 Downloading LVIS dataset (official Ultralytics method)...")
        
        try:
            # Official LVIS URLs from Ultralytics YAML
            labels_url = f"{ASSETS_URL}/lvis-labels-segments.zip"
            labels_path = self.lvis_dir / "lvis-labels-segments.zip"
            
            # Download LVIS labels
            if not labels_path.exists():
                print("Downloading LVIS labels...")
                if ULTRALYTICS_AVAILABLE:
                    download([labels_url], dir=self.lvis_dir.parent)
                else:
                    if not self.download_file_fallback(labels_url, labels_path):
                        return []
            
            # Download LVIS annotations (separate from labels)
            lvis_ann_url = "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip"
            lvis_ann_path = self.lvis_dir / "lvis_v1_val.json.zip"
            
            if not lvis_ann_path.exists():
                print("Downloading LVIS annotations...")
                if ULTRALYTICS_AVAILABLE:
                    download([lvis_ann_url], dir=self.lvis_dir)
                else:
                    if not self.download_file_fallback(lvis_ann_url, lvis_ann_path):
                        return []
                
                print("Extracting LVIS annotations...")
                with zipfile.ZipFile(lvis_ann_path, 'r') as zip_ref:
                    zip_ref.extractall(self.lvis_dir)
            
            # Load LVIS annotations
            lvis_json_path = self.lvis_dir / "lvis_v1_val.json"
            if not lvis_json_path.exists():
                print("❌ LVIS annotations not found!")
                return []
            
            print("Loading LVIS annotations...")
            with open(lvis_json_path, 'r') as f:
                lvis_data = json.load(f)
            
            # LVIS has 1203 categories
            categories = {cat['id']: cat['name'] for cat in lvis_data['categories']}
            print(f"Loaded {len(categories)} LVIS categories")
            
            # Group annotations by image (process in batches for memory efficiency)
            image_annotations = {}
            processed_annotations = 0
            max_annotations = 30000  # Process more for better coverage
            
            for ann in lvis_data['annotations']:
                if processed_annotations >= max_annotations:
                    break
                image_id = ann['image_id']
                if image_id not in image_annotations:
                    image_annotations[image_id] = []
                image_annotations[image_id].append(ann)
                processed_annotations += 1
            
            print(f"Processed {processed_annotations} annotations for {len(image_annotations)} images")
            
            # Select random images
            images_with_annotations = [img for img in lvis_data['images'] 
                                     if img['id'] in image_annotations]
            num_to_select = min(num_images, len(images_with_annotations))
            selected_images = random.sample(images_with_annotations, num_to_select)
            
            print(f"Selected {len(selected_images)} LVIS images to download")
            
            # Download images (LVIS uses COCO images)
            images_dir = self.lvis_dir / "images" / "val2017"
            images_dir.mkdir(exist_ok=True, parents=True)
            
            # Official COCO image URLs from Ultralytics LVIS YAML
            image_urls = []
            for img_info in selected_images:
                image_path = images_dir / img_info['file_name']
                if not image_path.exists():
                    image_url = f"http://images.cocodataset.org/val2017/{img_info['file_name']}"
                    image_urls.append(image_url)
            
            if image_urls:
                if ULTRALYTICS_AVAILABLE:
                    print("Downloading LVIS images (batch)...")
                    download(image_urls, dir=images_dir, threads=4)
                else:
                    print("Downloading LVIS images (sequential)...")
                    for url in tqdm(image_urls, desc="Downloading"):
                        filename = Path(url).name
                        self.download_file_fallback(url, images_dir / filename)
            
            # Process images
            dataset_images = []
            for img_info in selected_images:
                image_path = images_dir / img_info['file_name']
                
                if image_path.exists():
                    # Get ground truth objects
                    ground_truth_objects = []
                    if img_info['id'] in image_annotations:
                        for ann in image_annotations[img_info['id']]:
                            if ann['category_id'] in categories:
                                ground_truth_objects.append(categories[ann['category_id']])
                    
                    if ground_truth_objects:
                        dataset_images.append({
                            'dataset': 'LVIS',
                            'image_path': str(image_path),
                            'image_id': img_info['id'],
                            'ground_truth_objects': ground_truth_objects,
                            'annotations': image_annotations.get(img_info['id'], [])
                        })
            
            print(f"✅ Successfully processed {len(dataset_images)} LVIS images")
            return dataset_images
            
        except Exception as e:
            print(f"❌ LVIS download failed: {e}")
            return []

class OfficialOpenImagesDownloader(OfficialDatasetDownloader):
    """Official Open Images downloader using FiftyOne (Ultralytics recommended)."""
    
    def __init__(self, base_dir: str = "datasets"):
        super().__init__(base_dir)
        self.oi_dir = self.base_dir / "open-images-v7"
        self.oi_dir.mkdir(exist_ok=True)
        
    def check_fiftyone_available(self) -> bool:
        """Check if FiftyOne is available."""
        try:
            import fiftyone as fo
            import fiftyone.zoo as foz
            return True
        except ImportError:
            return False
    
    def download_open_images_sample(self, num_images: int = 100) -> List[Dict[str, Any]]:
        """Download Open Images using FiftyOne (official Ultralytics method)."""
        print("🔄 Downloading Open Images v7 (official Ultralytics method)...")
        
        if not self.check_fiftyone_available():
            print("❌ FiftyOne not available. Install with: pip install fiftyone")
            print("   FiftyOne is required for official Open Images v7 downloads")
            return []
        
        try:
            import fiftyone as fo
            import fiftyone.zoo as foz
            
            print("Using FiftyOne for Open Images v7 download...")
            
            # Set dataset directory
            fo.config.dataset_zoo_dir = self.oi_dir.parent / "fiftyone" / "open-images-v7"
            
            # Load Open Images dataset (validation split)
            print("Loading Open Images v7 validation dataset...")
            dataset = foz.load_zoo_dataset(
                "open-images-v7",
                split="validation",
                label_types=["detections"],
                max_samples=num_images,
            )
            
            print(f"Loaded {len(dataset)} Open Images samples")
            
            # Process the downloaded dataset
            dataset_images = []
            
            for sample in tqdm(dataset, desc="Processing Open Images"):
                if sample.ground_truth and sample.ground_truth.detections:
                    ground_truth_objects = [det.label for det in sample.ground_truth.detections]
                    
                    if ground_truth_objects and os.path.exists(sample.filepath):
                        dataset_images.append({
                            'dataset': 'Open Images v7',
                            'image_path': sample.filepath,
                            'image_id': sample.id,
                            'ground_truth_objects': ground_truth_objects,
                            'annotations': [det.label for det in sample.ground_truth.detections]
                        })
            
            print(f"✅ Successfully processed {len(dataset_images)} Open Images v7 images")
            return dataset_images
            
        except Exception as e:
            print(f"❌ Open Images v7 download failed: {e}")
            print("   Try installing FiftyOne: pip install fiftyone")
            return []

class OfficialDatasetComparison:
    """Official dataset comparison using Ultralytics methods."""
    
    def __init__(self, model_path: str = "yolo11n_object365.pt"):
        """Initialize with official Ultralytics-based downloaders."""
        print("="*70)
        print("OFFICIAL ULTRALYTICS DATASET COMPARISON")
        print("="*70)
        print(f"Using model: {model_path}")
        
        if ULTRALYTICS_AVAILABLE:
            print("✅ Ultralytics utilities available - using official methods")
        else:
            print("⚠️  Ultralytics utilities not available - using fallback methods")
        
        try:
            self.detector = ObjectDetectionWithDistanceAngle(
                model_path=model_path,
                verbose=False
            )
            print("✅ Object detection system initialized")
            print(f"Model classes available: {len(self.detector.model.names)}")
        except Exception as e:
            print(f"❌ Failed to initialize detection system: {e}")
            raise
        
        # Initialize official downloaders
        self.downloaders = {
            'COCO': OfficialCOCODownloader(),
            'PASCAL VOC': OfficialPASCALVOCDownloader(),
            'LVIS': OfficialLVISDownloader(),
            'Open Images': OfficialOpenImagesDownloader()
        }
    
    def download_all_datasets(self, images_per_dataset: int = 100) -> List[Dict[str, Any]]:
        """Download from all datasets using official methods."""
        print(f"\nDownloading {images_per_dataset} images from each dataset using official methods...")
        print("=" * 70)
        
        all_images = []
        dataset_results = {}
        
        # Download COCO (80 classes)
        print("\n🔄 Processing COCO Dataset (80 classes)...")
        try:
            coco_images = self.downloaders['COCO'].download_coco_sample(images_per_dataset)
            all_images.extend(coco_images)
            dataset_results['COCO'] = len(coco_images)
        except Exception as e:
            print(f"❌ COCO failed: {e}")
            dataset_results['COCO'] = 0
        
        # Download PASCAL VOC (20 classes)
        print("\n🔄 Processing PASCAL VOC Dataset (20 classes)...")
        try:
            voc_images = self.downloaders['PASCAL VOC'].download_pascal_voc_sample(images_per_dataset)
            all_images.extend(voc_images)
            dataset_results['PASCAL VOC'] = len(voc_images)
        except Exception as e:
            print(f"❌ PASCAL VOC failed: {e}")
            dataset_results['PASCAL VOC'] = 0
        
        # Download LVIS (1203 classes)
        print("\n🔄 Processing LVIS Dataset (1203 classes)...")
        try:
            lvis_images = self.downloaders['LVIS'].download_lvis_sample(images_per_dataset)
            all_images.extend(lvis_images)
            dataset_results['LVIS'] = len(lvis_images)
        except Exception as e:
            print(f"❌ LVIS failed: {e}")
            dataset_results['LVIS'] = 0
        
        # Download Open Images (601 classes) - requires FiftyOne
        print("\n🔄 Processing Open Images v7 Dataset (601 classes)...")
        try:
            oi_images = self.downloaders['Open Images'].download_open_images_sample(images_per_dataset)
            all_images.extend(oi_images)
            dataset_results['Open Images v7'] = len(oi_images)
        except Exception as e:
            print(f"❌ Open Images v7 failed: {e}")
            dataset_results['Open Images v7'] = 0
        
        # Summary
        print("\n" + "=" * 70)
        print("OFFICIAL ULTRALYTICS DATASET DOWNLOAD SUMMARY")
        print("=" * 70)
        for dataset, count in dataset_results.items():
            status = "✅" if count > 0 else "❌"
            print(f"{status} {dataset}: {count} images")
        
        total_images = len(all_images)
        print(f"\nTotal images downloaded: {total_images}")
        
        if total_images > 0:
            print(f"✅ Success! Downloaded from {sum(1 for count in dataset_results.values() if count > 0)} datasets")
        else:
            print("❌ No images downloaded! Check internet connection and dependencies.")
        
        print("=" * 70)
        return all_images
    
    def run_detection_on_image(self, image_path: str) -> List[str]:
        """Run object detection on a single image."""
        try:
            frame = cv2.imread(image_path)
            if frame is None:
                return []
            
            result = self.detector.process_single_frame(frame, enable_tracking=False)
            
            detected_objects = []
            for detection in result['detections']:
                class_name = detection.get('class', detection.get('class_name', 'unknown'))
                detected_objects.append(class_name)
            
            return detected_objects
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return []
    
    def compare_datasets(self, dataset_images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compare detection results with ground truth."""
        print("Running official Ultralytics-based object detection and comparison...")
        
        comparison_results = []
        
        for img_data in tqdm(dataset_images, desc="Processing images"):
            image_path = img_data['image_path']
            
            if not os.path.exists(image_path):
                continue
            
            detected_objects = self.run_detection_on_image(image_path)
            ground_truth_objects = img_data['ground_truth_objects']
            
            result = {
                'dataset': img_data['dataset'],
                'image_id': img_data['image_id'],
                'image_path': image_path,
                'ground_truth_objects': ground_truth_objects,
                'detected_objects': detected_objects,
                'ground_truth_count': len(ground_truth_objects),
                'detected_count': len(detected_objects),
                'ground_truth_unique': list(set(ground_truth_objects)),
                'detected_unique': list(set(detected_objects)),
                'ground_truth_unique_count': len(set(ground_truth_objects)),
                'detected_unique_count': len(set(detected_objects))
            }
            
            comparison_results.append(result)
        
        return comparison_results
    
    def generate_csv_report(self, comparison_results: List[Dict[str, Any]], 
                          output_file: str = "official_ultralytics_comparison.csv"):
        """Generate comprehensive CSV report."""
        print(f"Generating official Ultralytics CSV report: {output_file}")
        
        csv_data = []
        for result in comparison_results:
            base_row = {
                'Dataset': result['dataset'],
                'Image_ID': result['image_id'],
                'Image_Path': result['image_path'],
                'Ground_Truth_Count': result['ground_truth_count'],
                'Detected_Count': result['detected_count'],
                'Ground_Truth_Unique_Count': result['ground_truth_unique_count'],
                'Detected_Unique_Count': result['detected_unique_count'],
                'Ground_Truth_Objects': '; '.join(result['ground_truth_objects']),
                'Detected_Objects': '; '.join(result['detected_objects']),
                'Ground_Truth_Unique': '; '.join(result['ground_truth_unique']),
                'Detected_Unique': '; '.join(result['detected_unique'])
            }
            csv_data.append(base_row)
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(output_file, index=False)
            print(f"✅ Official Ultralytics CSV report saved: {output_file}")
            
            # Print comprehensive summary
            print(f"\n" + "="*70)
            print("OFFICIAL ULTRALYTICS COMPARISON SUMMARY")
            print("="*70)
            print(f"Total images processed: {len(df)}")
            
            for dataset in df['Dataset'].unique():
                dataset_df = df[df['Dataset'] == dataset]
                count = len(dataset_df)
                avg_gt = dataset_df['Ground_Truth_Count'].mean()
                avg_det = dataset_df['Detected_Count'].mean()
                avg_gt_unique = dataset_df['Ground_Truth_Unique_Count'].mean()
                avg_det_unique = dataset_df['Detected_Unique_Count'].mean()
                
                print(f"\n{dataset}:")
                print(f"  Images: {count}")
                print(f"  Avg objects per image: GT={avg_gt:.1f}, Detected={avg_det:.1f}")
                print(f"  Avg unique classes per image: GT={avg_gt_unique:.1f}, Detected={avg_det_unique:.1f}")
            
            # Overall statistics
            overall_avg_gt = df['Ground_Truth_Count'].mean()
            overall_avg_det = df['Detected_Count'].mean()
            
            print(f"\nOverall Statistics:")
            print(f"  Average objects per image: GT={overall_avg_gt:.1f}, Detected={overall_avg_det:.1f}")
            print(f"  Detection rate: {(overall_avg_det/overall_avg_gt*100):.1f}%")
            print("="*70)
        else:
            print("❌ No data to write to CSV")

def main():
    """Main function for official Ultralytics dataset comparison."""
    print("="*70)
    print("OFFICIAL ULTRALYTICS DATASET COMPARISON TOOL")
    print("="*70)
    print()
    print("This tool uses official Ultralytics dataset configurations for:")
    print("✅ COCO (80 classes)")
    print("✅ PASCAL VOC (20 classes)")  
    print("✅ LVIS (1203 classes)")
    print("⚠️  Open Images v7 (601 classes) - requires FiftyOne")
    print()
    
    # Check requirements and locate model
    clean_code_root = Path(__file__).parent.parent.parent
    model_path = clean_code_root / "models" / "yolo11n_object365.pt"
    
    # Try alternative model paths
    if not model_path.exists():
        model_path = clean_code_root / "models" / "yolo11n_object365.onnx"
    
    if not model_path.exists():
        print(f"❌ Error: Model not found in {clean_code_root / 'models'}/")
        print("Expected: yolo11n_object365.pt or yolo11n_object365.onnx")
        return 1
    
    print(f"✅ Using model: {model_path}")
    
    try:
        # Get user input
        images_per_dataset = input("Images per dataset (default: 75): ").strip()
        if not images_per_dataset:
            images_per_dataset = 75
        else:
            images_per_dataset = int(images_per_dataset)
        
        # Set default output path to test_results/csv_outputs
        test_results_dir = clean_code_root / "test_results" / "csv_outputs"
        test_results_dir.mkdir(parents=True, exist_ok=True)
        
        default_output = str(test_results_dir / "official_ultralytics_comparison.csv")
        output_file = input(f"Output CSV (default: {default_output}): ").strip()
        if not output_file:
            output_file = default_output
        
        print(f"\nStarting official Ultralytics comparison with {images_per_dataset} images per dataset...")
        
        # Initialize comparison
        comparison = OfficialDatasetComparison(model_path=model_path)
        
        # Download datasets
        dataset_images = comparison.download_all_datasets(images_per_dataset)
        
        if not dataset_images:
            print("❌ No images downloaded successfully!")
            print("\nTroubleshooting:")
            print("1. Check internet connection")
            print("2. For Open Images v7: pip install fiftyone")
            print("3. Some datasets are large - ensure sufficient disk space")
            return 1
        
        # Run comparison
        results = comparison.compare_datasets(dataset_images)
        
        # Generate report
        comparison.generate_csv_report(results, output_file)
        
        print("\n✅ Official Ultralytics comparison completed successfully!")
        print(f"📊 Results saved to: {output_file}")
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
