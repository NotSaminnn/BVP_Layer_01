"""
Custom Image Downloader for Hallucination Testing
Downloads images from Unsplash and Pexels for BLV-specific scenarios
"""

import os
import json
import random
import csv
from pathlib import Path
from typing import List, Dict
import urllib.request
import time
from tqdm import tqdm

class CustomImageDownloader:
    """Download custom images for BLV hallucination testing scenarios."""
    
    # Unsplash Source API (doesn't require API key for random images)
    UNSPLASH_RANDOM_URL = "https://source.unsplash.com/800x600/"
    
    # Image scenarios for BLV system testing
    SCENARIOS = {
        'document_reading': [
            'book', 'menu', 'receipt', 'sign', 'newspaper', 'letter', 
            'prescription', 'label', 'form', 'magazine'
        ],
        'low_light': [
            'night', 'dark+room', 'dimly+lit', 'candlelight', 'twilight',
            'evening', 'shadows', 'dusk'
        ],
        'person_recognition': [
            'portrait', 'face', 'group+photo', 'family', 'meeting',
            'conference', 'crowd', 'selfie'
        ],
        'navigation': [
            'crosswalk', 'stairs', 'hallway', 'doorway', 'intersection',
            'sidewalk', 'path', 'corridor'
        ],
        'object_location': [
            'kitchen+counter', 'desk+setup', 'table+setting', 'shelf',
            'workspace', 'organized+items', 'scattered+objects'
        ],
        'scene_description': [
            'park', 'cafe', 'library', 'store', 'restaurant',
            'waiting+room', 'classroom', 'lobby'
        ]
    }
    
    def __init__(self, output_dir: str, num_images_per_scenario: int = 5):
        """
        Initialize custom image downloader.
        
        Args:
            output_dir: Directory to save images
            num_images_per_scenario: Images per scenario (default: 5)
        """
        self.output_dir = Path(output_dir)
        self.num_images_per_scenario = num_images_per_scenario
        self.test_images_dir = self.output_dir / "test_images"
        self.test_images_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Output directory: {self.output_dir}")
        print(f"Images per scenario: {num_images_per_scenario}")
    
    def download_unsplash_image(self, keyword: str, save_path: Path, retry: int = 3) -> bool:
        """
        Download image from Unsplash Source API.
        
        Args:
            keyword: Search keyword
            save_path: Path to save image
            retry: Number of retries
        
        Returns:
            True if successful, False otherwise
        """
        url = f"{self.UNSPLASH_RANDOM_URL}?{keyword}"
        
        for attempt in range(retry):
            try:
                # Add delay to avoid rate limiting
                time.sleep(1)
                
                # Download image
                urllib.request.urlretrieve(url, str(save_path))
                
                # Verify file size (should be > 10KB)
                if save_path.stat().st_size > 10000:
                    return True
                else:
                    print(f"⚠️  Downloaded file too small: {save_path.name}")
                    save_path.unlink()
                    
            except Exception as e:
                print(f"⚠️  Attempt {attempt + 1} failed for '{keyword}': {e}")
                if save_path.exists():
                    save_path.unlink()
                time.sleep(2)
        
        return False
    
    def download_scenario_images(self):
        """Download images for all scenarios."""
        downloaded_images = []
        total_images = len(self.SCENARIOS) * self.num_images_per_scenario
        
        print(f"\n📥 Downloading {total_images} images from Unsplash...")
        print("⚠️  This may take 5-10 minutes (1-2 second delay per image to avoid rate limits)\n")
        
        pbar = tqdm(total=total_images, desc="Downloading images")
        
        for scenario_name, keywords in self.SCENARIOS.items():
            for i in range(self.num_images_per_scenario):
                # Select random keyword from scenario
                keyword = random.choice(keywords)
                
                # Generate unique filename
                image_id = f"custom_{scenario_name}_{i+1:02d}"
                save_path = self.test_images_dir / f"{image_id}.jpg"
                
                # Skip if already exists
                if save_path.exists():
                    pbar.update(1)
                    downloaded_images.append({
                        'image_id': image_id,
                        'scenario': scenario_name,
                        'keyword': keyword,
                        'path': save_path
                    })
                    continue
                
                # Download image
                success = self.download_unsplash_image(keyword, save_path)
                
                if success:
                    downloaded_images.append({
                        'image_id': image_id,
                        'scenario': scenario_name,
                        'keyword': keyword,
                        'path': save_path
                    })
                else:
                    print(f"❌ Failed to download: {image_id} ({keyword})")
                
                pbar.update(1)
        
        pbar.close()
        
        print(f"\n✅ Downloaded {len(downloaded_images)} images")
        return downloaded_images
    
    def create_manual_annotation_template(self, downloaded_images: List[Dict]):
        """
        Create CSV template for manual annotations.
        
        Args:
            downloaded_images: List of downloaded image metadata
        """
        csv_path = self.output_dir / 'custom_images_annotation_template.csv'
        
        csv_data = []
        for img_info in downloaded_images:
            csv_data.append({
                'image_id': img_info['image_id'],
                'image_path': f"test_images/{img_info['image_id']}.jpg",
                'scenario': img_info['scenario'],
                'keyword': img_info['keyword'],
                'scene_type': img_info['scenario'],
                'objects_present': 'TODO: Add comma-separated objects',
                'objects_count': '0',
                'objects_absent': 'TODO: Add comma-separated absent objects',
                'ocr_text': 'TODO: Add text if document, else null',
                'persons_present': 'TODO: Add count or null',
                'lighting_condition': 'low_light' if img_info['scenario'] == 'low_light' else 'normal',
                'source': 'unsplash'
            })
        
        # Save template
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['image_id', 'image_path', 'scenario', 'keyword', 'scene_type', 
                         'objects_present', 'objects_count', 'objects_absent', 'ocr_text', 
                         'persons_present', 'lighting_condition', 'source']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
        
        print(f"\n✅ Annotation template saved: {csv_path}")
        print(f"   Total images: {len(csv_data)}")
        print("\n⚠️  MANUAL ANNOTATION REQUIRED:")
        print("   1. Open custom_images_annotation_template.csv")
        print("   2. View each image and fill in:")
        print("      - objects_present: Comma-separated list (e.g., 'chair,desk,laptop')")
        print("      - objects_count: Number of objects")
        print("      - objects_absent: Objects that could be there but aren't")
        print("      - ocr_text: Text visible in document images (or 'null')")
        print("      - persons_present: Number of people (or 'null')")
        print("   3. Save as 'custom_images_annotations.csv'")
        
        return csv_path
    
    def merge_with_coco_annotations(self, custom_annotations_path: str):
        """
        Merge custom annotations with COCO ground truth CSV.
        
        Args:
            custom_annotations_path: Path to custom annotations CSV
        """
        coco_csv = self.output_dir / 'ground_truth_annotations.csv'
        custom_csv = Path(custom_annotations_path)
        merged_csv = self.output_dir / 'ground_truth_annotations_merged.csv'
        
        if not custom_csv.exists():
            print(f"❌ Custom annotations not found: {custom_csv}")
            print("   Please complete manual annotation first!")
            return
        
        # Load both CSVs
        import pandas as pd
        
        coco_df = pd.read_csv(coco_csv) if coco_csv.exists() else pd.DataFrame()
        custom_df = pd.read_csv(custom_csv)
        
        # Merge
        merged_df = pd.concat([coco_df, custom_df], ignore_index=True)
        merged_df.to_csv(merged_csv, index=False)
        
        print(f"\n✅ Merged annotations saved: {merged_csv}")
        print(f"   COCO images: {len(coco_df)}")
        print(f"   Custom images: {len(custom_df)}")
        print(f"   Total: {len(merged_df)}")
    
    def generate_statistics(self, csv_path: Path):
        """Generate dataset statistics."""
        import pandas as pd
        
        if not csv_path.exists():
            print(f"❌ CSV not found: {csv_path}")
            return
        
        df = pd.read_csv(csv_path)
        
        print("\n" + "=" * 70)
        print("CUSTOM DATASET STATISTICS")
        print("=" * 70)
        
        print(f"\nTotal images: {len(df)}")
        print(f"\nScenario distribution:")
        print(df['scenario'].value_counts().to_string())
        
        print(f"\nSource distribution:")
        print(df['source'].value_counts().to_string())
        
        print("\n" + "=" * 70)
    
    def run_full_pipeline(self):
        """Execute complete download pipeline."""
        print("\n" + "=" * 70)
        print("CUSTOM IMAGE DOWNLOADER FOR HALLUCINATION TESTING")
        print("=" * 70 + "\n")
        
        # Step 1: Download images
        downloaded_images = self.download_scenario_images()
        
        # Step 2: Create annotation template
        template_path = self.create_manual_annotation_template(downloaded_images)
        
        # Step 3: Generate statistics
        self.generate_statistics(template_path)
        
        print("\n✅ Pipeline complete!")
        print(f"📁 Images: {self.test_images_dir}")
        print(f"📄 Annotation template: {template_path}")
        print("\n⚠️  NEXT STEPS:")
        print("   1. Manually annotate images in 'custom_images_annotation_template.csv'")
        print("   2. Save as 'custom_images_annotations.csv'")
        print("   3. Run merge script to combine with COCO annotations")


def main():
    """Main function to run custom image downloader."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download custom images for hallucination testing")
    parser.add_argument('--output_dir', type=str, 
                       default='e:/BVP_LAYER01 (2)/BVP_LAYER01/hallucination_testing',
                       help='Output directory for images and annotations')
    parser.add_argument('--num_images', type=int, default=5,
                       help='Number of images per scenario (default: 5)')
    parser.add_argument('--merge', type=str, default=None,
                       help='Path to custom annotations CSV for merging with COCO')
    
    args = parser.parse_args()
    
    downloader = CustomImageDownloader(args.output_dir, args.num_images)
    
    if args.merge:
        downloader.merge_with_coco_annotations(args.merge)
    else:
        downloader.run_full_pipeline()


if __name__ == "__main__":
    main()
