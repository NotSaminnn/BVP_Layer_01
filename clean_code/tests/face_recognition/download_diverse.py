"""
Automated Face Dataset Downloader
Downloads diverse face datasets for comprehensive face recognition evaluation.
Supports: LFW, Synthetic faces, and organized unknown persons.
"""

import os
import sys
import time
import random
import shutil
import tarfile
import requests
from pathlib import Path
from tqdm import tqdm
import urllib.request

class DiverseFaceDatasetDownloader:
    def __init__(self, output_dir="test_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        print(f"Output directory: {self.output_dir.absolute()}")
        
    def download_file(self, url, output_path, desc="Downloading"):
        """Download file with progress bar"""
        try:
            response = requests.get(url, stream=True, timeout=30)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f, tqdm(
                total=total_size, unit='B', unit_scale=True, desc=desc
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            return True
        except Exception as e:
            print(f"  ✗ Error downloading: {e}")
            return False
    
    def download_synthetic_face(self, save_path, max_retries=5):
        """Download one synthetic face from ThisPersonDoesNotExist"""
        for attempt in range(max_retries):
            try:
                # ThisPersonDoesNotExist.com generates new face each time
                url = "https://thispersondoesnotexist.com/"
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    with open(save_path, 'wb') as f:
                        f.write(response.content)
                    return True
                    
                time.sleep(2)  # Rate limiting
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(3)
                    continue
                else:
                    print(f"  ✗ Failed after {max_retries} attempts: {e}")
                    return False
        
        return False
    
    def download_synthetic_person(self, person_name, num_photos=40):
        """Download multiple synthetic faces for one person"""
        person_dir = self.output_dir / person_name
        person_dir.mkdir(exist_ok=True)
        
        print(f"\n📥 Downloading {num_photos} synthetic faces for {person_name}...")
        
        successful = 0
        for i in tqdm(range(num_photos), desc="Progress"):
            save_path = person_dir / f"photo_{i+1:03d}.jpg"
            
            if self.download_synthetic_face(save_path):
                successful += 1
            
            # Rate limiting - be nice to the server
            time.sleep(random.uniform(1.5, 3.0))
        
        print(f"  ✓ Successfully downloaded {successful}/{num_photos} photos")
        return successful
    
    def download_lfw_dataset(self, target_persons=10, photos_per_person=40):
        """Download and extract LFW dataset, select diverse persons"""
        print("\n" + "="*60)
        print("DOWNLOADING LFW DATASET (Labeled Faces in the Wild)")
        print("="*60)
        print("This may take 10-20 minutes depending on connection...")
        
        lfw_url = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
        temp_file = "lfw_temp.tgz"
        
        # Download LFW
        print("\n[1/4] Downloading LFW archive (~170 MB)...")
        if not self.download_file(lfw_url, temp_file, "Downloading LFW"):
            print("  ✗ Failed to download LFW")
            return 0
        
        # Extract
        print("\n[2/4] Extracting archive...")
        try:
            with tarfile.open(temp_file, 'r:gz') as tar:
                tar.extractall('lfw_temp')
            print("  ✓ Extraction complete")
        except Exception as e:
            print(f"  ✗ Extraction failed: {e}")
            return 0
        
        # Find persons with enough photos
        print("\n[3/4] Analyzing dataset...")
        lfw_dir = Path('lfw_temp/lfw')
        all_persons = [p for p in lfw_dir.iterdir() if p.is_dir()]
        
        # Filter persons with enough photos
        qualified_persons = []
        for person_dir in all_persons:
            photo_count = len(list(person_dir.glob('*.jpg')))
            if photo_count >= photos_per_person:
                qualified_persons.append((person_dir, photo_count))
        
        print(f"  Found {len(qualified_persons)} persons with {photos_per_person}+ photos")
        
        # Select randomly
        if len(qualified_persons) < target_persons:
            print(f"  ⚠ Only {len(qualified_persons)} persons available")
            target_persons = len(qualified_persons)
        
        selected = random.sample(qualified_persons, target_persons)
        
        # Copy to output directory
        print(f"\n[4/4] Copying {target_persons} persons...")
        copied = 0
        
        for idx, (person_dir, photo_count) in enumerate(selected, 3):  # Start from person_03
            new_name = f"person_{idx:02d}_{person_dir.name.replace(' ', '_')}"
            target_dir = self.output_dir / new_name
            target_dir.mkdir(exist_ok=True)
            
            # Copy photos
            photos = sorted(list(person_dir.glob('*.jpg')))[:photos_per_person]
            
            for photo in photos:
                try:
                    shutil.copy(photo, target_dir / photo.name)
                except Exception as e:
                    print(f"  ⚠ Error copying {photo.name}: {e}")
            
            actual_count = len(list(target_dir.glob('*.jpg')))
            print(f"  ✓ {new_name}: {actual_count} photos")
            copied += 1
        
        # Cleanup
        print("\n[Cleanup] Removing temporary files...")
        try:
            shutil.rmtree('lfw_temp')
            os.remove(temp_file)
            print("  ✓ Cleanup complete")
        except Exception as e:
            print(f"  ⚠ Cleanup warning: {e}")
        
        return copied
    
    def create_unknown_persons(self, num_persons=10, photos_per_person=20):
        """Create unknown persons dataset using synthetic faces"""
        print("\n" + "="*60)
        print(f"CREATING UNKNOWN PERSONS DATASET")
        print("="*60)
        print(f"Generating {num_persons} unknown persons...")
        
        unknown_base = self.output_dir / "unknown_persons"
        unknown_base.mkdir(exist_ok=True)
        
        created = 0
        for i in range(1, num_persons + 1):
            person_name = f"unknown_{i:02d}"
            person_dir = unknown_base / person_name
            person_dir.mkdir(exist_ok=True)
            
            print(f"\n📥 Creating {person_name} ({photos_per_person} photos)...")
            
            successful = 0
            for j in tqdm(range(photos_per_person), desc="Progress"):
                save_path = person_dir / f"photo_{j+1:03d}.jpg"
                
                if self.download_synthetic_face(save_path):
                    successful += 1
                
                time.sleep(random.uniform(1.5, 3.0))
            
            print(f"  ✓ {person_name}: {successful}/{photos_per_person} photos")
            if successful > 0:
                created += 1
        
        return created
    
    def copy_existing_persons(self):
        """Copy Maliha and Reshad from enrolled_persons to test_dataset"""
        print("\n" + "="*60)
        print("COPYING EXISTING PERSONS (Maliha, Reshad)")
        print("="*60)
        
        enrolled_dir = Path("enrolled_persons")
        if not enrolled_dir.exists():
            print("  ⚠ enrolled_persons directory not found")
            return 0
        
        copied = 0
        for idx, folder_name in enumerate(["Maliha_photos", "Reshad_photos"], 1):
            source_dir = enrolled_dir / folder_name
            
            if not source_dir.exists():
                print(f"  ⚠ {folder_name} not found")
                continue
            
            person_name = folder_name.replace("_photos", "").lower()
            target_name = f"person_{idx:02d}_{person_name}"
            target_dir = self.output_dir / target_name
            
            if target_dir.exists():
                print(f"  ℹ {target_name} already exists, skipping")
                copied += 1
                continue
            
            try:
                shutil.copytree(source_dir, target_dir)
                photo_count = len(list(target_dir.glob('*.jpg')))
                print(f"  ✓ Copied {target_name}: {photo_count} photos")
                copied += 1
            except Exception as e:
                print(f"  ✗ Error copying {folder_name}: {e}")
        
        return copied
    
    def generate_full_dataset(self):
        """Generate complete diverse dataset"""
        print("\n" + "="*70)
        print(" 🎯 DIVERSE FACE DATASET GENERATION")
        print("="*70)
        print(f"Target: 12 enrolled persons + 10 unknown persons")
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        start_time = time.time()
        
        # Step 1: Copy existing persons (Maliha, Reshad)
        print("\n" + "-"*70)
        print("STEP 1: Copying Existing Persons")
        print("-"*70)
        existing_count = self.copy_existing_persons()
        
        # Step 2: Download LFW persons
        print("-"*70)
        print("STEP 2: Downloading LFW Dataset")
        print("-"*70)
        lfw_count = self.download_lfw_dataset(target_persons=3, photos_per_person=40)
        
        # Step 3: Generate synthetic persons if needed
        total_enrolled = existing_count + lfw_count
        needed = 5 - total_enrolled
        
        if needed > 0:
            print("\n" + "-"*70)
            print(f"STEP 3: Generating {needed} Synthetic Persons")
            print("-"*70)
            
            for i in range(needed):
                person_num = total_enrolled + i + 1
                person_name = f"person_{person_num:02d}_synthetic_gen"
                self.download_synthetic_person(person_name, num_photos=40)
        
        # Step 4: Create unknown persons
        print("\n" + "-"*70)
        print("STEP 4: Creating Unknown Persons Dataset")
        print("-"*70)
        unknown_count = self.create_unknown_persons(num_persons=5, photos_per_person=20)
        
        # Summary
        elapsed = time.time() - start_time
        
        print("\n" + "="*70)
        print(" ✅ DATASET GENERATION COMPLETE")
        print("="*70)
        
        # Count actual results
        enrolled_dirs = sorted([d for d in self.output_dir.iterdir() 
                               if d.is_dir() and d.name.startswith("person_")])
        
        unknown_base = self.output_dir / "unknown_persons"
        unknown_dirs = sorted([d for d in unknown_base.iterdir() 
                              if d.is_dir()]) if unknown_base.exists() else []
        
        print(f"\n📊 Dataset Statistics:")
        print(f"  Duration: {elapsed/60:.1f} minutes")
        print(f"\n  Enrolled Persons: {len(enrolled_dirs)}")
        
        total_enrolled_photos = 0
        for person_dir in enrolled_dirs:
            count = len(list(person_dir.glob('*.jpg')))
            total_enrolled_photos += count
            print(f"    • {person_dir.name}: {count} photos")
        
        print(f"\n  Unknown Persons: {len(unknown_dirs)}")
        total_unknown_photos = 0
        for person_dir in unknown_dirs:
            count = len(list(person_dir.glob('*.jpg')))
            total_unknown_photos += count
            print(f"    • {person_dir.name}: {count} photos")
        
        print(f"\n📈 Total Statistics:")
        print(f"  Total persons: {len(enrolled_dirs) + len(unknown_dirs)}")
        print(f"  Total photos: {total_enrolled_photos + total_unknown_photos}")
        print(f"  Enrolled photos: {total_enrolled_photos}")
        print(f"  Unknown photos: {total_unknown_photos}")
        
        print(f"\n📁 Dataset location: {self.output_dir.absolute()}")
        print(f"\n✅ Ready for evaluation!")
        print("="*70)
        
        return len(enrolled_dirs), len(unknown_dirs)

def main():
    """Main execution"""
    print("\n" + "="*70)
    print(" 🚀 FACE RECOGNITION DATASET DOWNLOADER")
    print("="*70)
    print("This script will download a diverse face recognition dataset:")
    print("  • 2 existing persons (Maliha, Reshad)")
    print("  • 3 persons from LFW dataset or synthetic")
    print("  • 0-3 synthetic generated persons (if needed)")
    print("  • 5 unknown persons for FPR testing")
    print("\nTotal download size: ~200-300 MB")
    print("Estimated time: 30-60 minutes (depending on connection)")
    
    response = input("\n⚠ Proceed with download? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("\n❌ Download cancelled")
        return
    
    downloader = DiverseFaceDatasetDownloader(output_dir="test_dataset")
    
    try:
        enrolled, unknown = downloader.generate_full_dataset()
        
        if enrolled >= 5 and unknown >= 5:
            print("\n✅ SUCCESS: Dataset is ready for evaluation!")
            print(f"\nNext steps:")
            print(f"  1. Review dataset: test_dataset/")
            print(f"  2. Run evaluation: py automated_evaluation_multi_person.py")
            print(f"  3. Generate figures: py generate_ieee_figures.py")
        else:
            print("\n⚠ WARNING: Dataset may be incomplete")
            print(f"  Enrolled: {enrolled}/12")
            print(f"  Unknown: {unknown}/10")
            print(f"\nYou can still run evaluation, but results may be limited.")
    
    except KeyboardInterrupt:
        print("\n\n❌ Download interrupted by user")
        print("⚠ Partial dataset may exist in test_dataset/")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
