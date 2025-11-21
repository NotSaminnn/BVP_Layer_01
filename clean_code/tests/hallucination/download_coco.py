"""
COCO Dataset Downloader and Annotation Converter
Downloads COCO validation images and converts annotations to hallucination testing format
"""

import json
import os
import urllib.request
from pathlib import Path
import csv
from tqdm import tqdm
import random

class COCODownloader:
    """Download and convert COCO dataset for hallucination testing."""
    
    COCO_VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
    COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    
    def __init__(self, output_dir: str, num_images: int = 100):
        """
        Initialize COCO downloader.
        
        Args:
            output_dir: Directory to save images and annotations
            num_images: Number of images to download (default: 100)
        """
        self.output_dir = Path(output_dir)
        self.num_images = num_images
        self.test_images_dir = self.output_dir / "test_images"
        self.test_images_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Output directory: {self.output_dir}")
        print(f"Number of images to download: {num_images}")
    
    def download_annotations(self):
        """Download COCO annotations JSON file."""
        annotations_path = self.output_dir / "annotations_trainval2017.zip"
        
        if annotations_path.exists():
            print(f"✅ Annotations already exist: {annotations_path}")
            return annotations_path
        
        print(f"📥 Downloading COCO annotations...")
        urllib.request.urlretrieve(
            self.COCO_ANNOTATIONS_URL,
            str(annotations_path),
            reporthook=self._download_progress
        )
        print(f"\n✅ Downloaded: {annotations_path}")
        
        # Extract annotations
        print("📦 Extracting annotations...")
        import zipfile
        with zipfile.ZipFile(annotations_path, 'r') as zip_ref:
            zip_ref.extractall(self.output_dir)
        print("✅ Annotations extracted")
        
        return self.output_dir / "annotations" / "instances_val2017.json"
    
    def download_images(self):
        """Download COCO validation images."""
        images_zip_path = self.output_dir / "val2017.zip"
        
        if images_zip_path.exists():
            print(f"✅ Images zip already exists: {images_zip_path}")
        else:
            print(f"📥 Downloading COCO validation images (1GB+)...")
            print("⚠️  This may take 5-15 minutes depending on your connection")
            urllib.request.urlretrieve(
                self.COCO_VAL_IMAGES_URL,
                str(images_zip_path),
                reporthook=self._download_progress
            )
            print(f"\n✅ Downloaded: {images_zip_path}")
        
        # Extract images
        images_dir = self.output_dir / "val2017"
        if not images_dir.exists():
            print("📦 Extracting images...")
            import zipfile
            with zipfile.ZipFile(images_zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.output_dir)
            print("✅ Images extracted")
        else:
            print(f"✅ Images already extracted: {images_dir}")
        
        return images_dir
    
    def load_coco_annotations(self, annotations_path: Path):
        """Load COCO annotations JSON."""
        print(f"📂 Loading annotations from: {annotations_path}")
        with open(annotations_path, 'r') as f:
            coco_data = json.load(f)
        
        print(f"   Images: {len(coco_data['images'])}")
        print(f"   Annotations: {len(coco_data['annotations'])}")
        print(f"   Categories: {len(coco_data['categories'])}")
        
        return coco_data
    
    def select_diverse_images(self, coco_data: dict):
        """
        Select diverse images with varying object counts and scene types.
        
        Returns:
            List of selected image IDs
        """
        # Group images by number of objects
        image_object_counts = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            image_object_counts[img_id] = image_object_counts.get(img_id, 0) + 1
        
        # Sort by object count
        sorted_images = sorted(image_object_counts.items(), key=lambda x: x[1])
        
        # Select images with diverse object counts
        selected = []
        
        # Few objects (1-3): 20%
        few_obj = [img_id for img_id, count in sorted_images if 1 <= count <= 3]
        selected.extend(random.sample(few_obj, min(int(self.num_images * 0.2), len(few_obj))))
        
        # Medium objects (4-8): 50%
        med_obj = [img_id for img_id, count in sorted_images if 4 <= count <= 8]
        selected.extend(random.sample(med_obj, min(int(self.num_images * 0.5), len(med_obj))))
        
        # Many objects (9+): 30%
        many_obj = [img_id for img_id, count in sorted_images if count >= 9]
        selected.extend(random.sample(many_obj, min(int(self.num_images * 0.3), len(many_obj))))
        
        # Fill remaining with random
        remaining = self.num_images - len(selected)
        if remaining > 0:
            all_ids = list(image_object_counts.keys())
            available = [img_id for img_id in all_ids if img_id not in selected]
            selected.extend(random.sample(available, min(remaining, len(available))))
        
        print(f"✅ Selected {len(selected)} diverse images")
        return selected[:self.num_images]
    
    def convert_to_ground_truth_csv(self, coco_data: dict, selected_image_ids: list):
        """
        Convert COCO annotations to ground truth CSV format.
        
        Args:
            coco_data: COCO dataset JSON
            selected_image_ids: List of selected image IDs
        """
        # Create category mapping
        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        # Create image mapping
        images = {img['id']: img for img in coco_data['images']}
        
        # Group annotations by image
        image_annotations = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in selected_image_ids:
                continue
            if img_id not in image_annotations:
                image_annotations[img_id] = []
            image_annotations[img_id].append(ann)
        
        # Generate absent objects (objects that could be present but aren't)
        all_categories = set(categories.values())
        
        # Prepare CSV data
        csv_data = []
        
        for img_id in tqdm(selected_image_ids, desc="Converting annotations"):
            if img_id not in images:
                continue
            
            img_info = images[img_id]
            img_filename = img_info['file_name']
            
            # Get objects present in image
            objects_present = []
            if img_id in image_annotations:
                for ann in image_annotations[img_id]:
                    cat_name = categories[ann['category_id']]
                    objects_present.append(cat_name)
            
            # Remove duplicates and sort
            objects_present = sorted(set(objects_present))
            objects_present_str = ','.join(objects_present) if objects_present else "none"
            
            # Generate absent objects (sample 5-10 objects not in image)
            present_set = set(objects_present)
            absent_candidates = list(all_categories - present_set)
            num_absent = min(random.randint(5, 10), len(absent_candidates))
            objects_absent = random.sample(absent_candidates, num_absent)
            objects_absent_str = ','.join(sorted(objects_absent))
            
            # Determine scene type based on objects
            scene_type = self._infer_scene_type(objects_present)
            
            # Copy image to test_images directory
            src_path = self.output_dir / "val2017" / img_filename
            dst_path = self.test_images_dir / f"coco_{img_id:012d}.jpg"
            
            if src_path.exists():
                import shutil
                shutil.copy2(src_path, dst_path)
            
            csv_data.append({
                'image_id': f"coco_{img_id:012d}",
                'image_path': f"test_images/coco_{img_id:012d}.jpg",
                'scene_type': scene_type,
                'objects_present': objects_present_str,
                'objects_count': len(objects_present),
                'objects_absent': objects_absent_str,
                'ocr_text': 'null',
                'persons_present': 'null',
                'lighting_condition': 'normal',
                'source': 'coco_val2017'
            })
        
        # Save to CSV
        csv_path = self.output_dir / 'ground_truth_annotations.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['image_id', 'image_path', 'scene_type', 'objects_present', 
                         'objects_count', 'objects_absent', 'ocr_text', 'persons_present', 
                         'lighting_condition', 'source']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
        
        print(f"\n✅ Ground truth CSV saved: {csv_path}")
        print(f"   Total images: {len(csv_data)}")
        print(f"   Copied to: {self.test_images_dir}")
        
        return csv_path
    
    def _infer_scene_type(self, objects: list) -> str:
        """Infer scene type from objects present."""
        objects_set = set(obj.lower() for obj in objects)
        
        # Kitchen indicators
        kitchen_objects = {'refrigerator', 'microwave', 'oven', 'sink', 'toaster', 
                          'knife', 'fork', 'spoon', 'bowl', 'cup', 'bottle'}
        if len(objects_set & kitchen_objects) >= 2:
            return 'kitchen'
        
        # Living room indicators
        living_room_objects = {'couch', 'tv', 'remote', 'chair', 'book', 'vase'}
        if len(objects_set & living_room_objects) >= 2:
            return 'living_room'
        
        # Office indicators
        office_objects = {'laptop', 'keyboard', 'mouse', 'monitor', 'desk', 'chair'}
        if len(objects_set & office_objects) >= 2:
            return 'office'
        
        # Outdoor indicators
        outdoor_objects = {'car', 'bicycle', 'motorcycle', 'bus', 'truck', 'traffic light', 
                          'bench', 'bird', 'cat', 'dog', 'horse'}
        if len(objects_set & outdoor_objects) >= 2:
            return 'outdoor'
        
        # Default
        return 'general'
    
    def _download_progress(self, block_num, block_size, total_size):
        """Display download progress."""
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        bar_length = 50
        filled = int(bar_length * percent / 100)
        bar = '█' * filled + '-' * (bar_length - filled)
        print(f'\r[{bar}] {percent:.1f}% ({downloaded / 1024 / 1024:.1f}MB / {total_size / 1024 / 1024:.1f}MB)', end='')
    
    def generate_statistics(self, csv_path: Path):
        """Generate dataset statistics."""
        import pandas as pd
        
        df = pd.read_csv(csv_path)
        
        print("\n" + "=" * 70)
        print("DATASET STATISTICS")
        print("=" * 70)
        
        print(f"\nTotal images: {len(df)}")
        print(f"\nScene type distribution:")
        print(df['scene_type'].value_counts().to_string())
        
        print(f"\nObject count distribution:")
        print(df['objects_count'].describe().to_string())
        
        print(f"\nMean objects per image: {df['objects_count'].mean():.2f}")
        print(f"Median objects per image: {df['objects_count'].median():.0f}")
        
        print("\n" + "=" * 70)
    
    def run_full_pipeline(self):
        """Execute complete download and conversion pipeline."""
        print("\n" + "=" * 70)
        print("COCO DATASET DOWNLOADER FOR HALLUCINATION TESTING")
        print("=" * 70 + "\n")
        
        # Step 1: Download annotations
        annotations_path = self.download_annotations()
        if not annotations_path.exists():
            annotations_path = self.output_dir / "annotations" / "instances_val2017.json"
        
        # Step 2: Download images
        images_dir = self.download_images()
        
        # Step 3: Load annotations
        coco_data = self.load_coco_annotations(annotations_path)
        
        # Step 4: Select diverse images
        selected_image_ids = self.select_diverse_images(coco_data)
        
        # Step 5: Convert to CSV
        csv_path = self.convert_to_ground_truth_csv(coco_data, selected_image_ids)
        
        # Step 6: Generate statistics
        self.generate_statistics(csv_path)
        
        print("\n✅ Pipeline complete!")
        print(f"📁 Images: {self.test_images_dir}")
        print(f"📄 Annotations: {csv_path}")


def main():
    """Main function to run COCO downloader."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download COCO dataset for hallucination testing")
    parser.add_argument('--output_dir', type=str, 
                       default='e:/BVP_LAYER01 (2)/BVP_LAYER01/hallucination_testing',
                       help='Output directory for images and annotations')
    parser.add_argument('--num_images', type=int, default=100,
                       help='Number of images to download (default: 100)')
    
    args = parser.parse_args()
    
    downloader = COCODownloader(args.output_dir, args.num_images)
    downloader.run_full_pipeline()


if __name__ == "__main__":
    main()
