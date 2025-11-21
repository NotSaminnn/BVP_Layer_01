"""
Download test photos for face recognition evaluation.
Downloads diverse face images for testing purposes.
"""

import os
import urllib.request
import json
from pathlib import Path

def download_test_images():
    """
    Download sample face images for testing.
    Using publicly available test datasets.
    """
    
    base_dir = Path(__file__).parent
    
    # Create person directories
    persons = {
        "Person1": base_dir / "enrolled_persons" / "Person1_photos",
        "Person2": base_dir / "enrolled_persons" / "Person2_photos",
        "Person3": base_dir / "enrolled_persons" / "Person3_photos",
        "Person4": base_dir / "enrolled_persons" / "Person4_photos",
        "Person5": base_dir / "enrolled_persons" / "Person5_photos",
    }
    
    unknown_dir = base_dir / "unknown_persons"
    
    # Create directories
    for person_dir in persons.values():
        person_dir.mkdir(parents=True, exist_ok=True)
    unknown_dir.mkdir(parents=True, exist_ok=True)
    
    print("✓ Directories created")
    
    # Instructions for manual download
    print("\n" + "="*60)
    print("📥 PHOTO DOWNLOAD INSTRUCTIONS")
    print("="*60)
    
    print("\n🔹 ENROLLED PERSONS (For Testing Accuracy):")
    print("\nFor each person (Person1-5), collect 20-30 photos:")
    print("  • Use Google Images, Unsplash, or Pexels")
    print("  • Search for: 'professional headshot', 'portrait photo', 'face photo'")
    print("  • Get variety: different angles, lighting, expressions")
    print("  • Save to: enrolled_persons/PersonX_photos/")
    
    print("\n📁 Folder structure:")
    for name, path in persons.items():
        print(f"  {path.relative_to(base_dir)}/")
        print(f"    ├── 001.jpg")
        print(f"    ├── 002.jpg")
        print(f"    ├── ... (20-30 images)")
    
    print("\n🔹 UNKNOWN PERSONS (For Testing False Positives):")
    print("\nCollect 30-50 photos of DIFFERENT people:")
    print("  • People NOT in the enrolled set")
    print("  • Similar photo conditions")
    print("  • Save to: unknown_persons/")
    
    print(f"\n📁 {unknown_dir.relative_to(base_dir)}/")
    print("    ├── unknown_001.jpg")
    print("    ├── unknown_002.jpg")
    print("    ├── ... (30-50 images)")
    
    print("\n" + "="*60)
    print("🌐 RECOMMENDED FREE PHOTO SOURCES:")
    print("="*60)
    print("  • Unsplash.com - Free high-quality photos")
    print("  • Pexels.com - Free stock photos")
    print("  • Pixabay.com - Free images")
    print("  • Google Images (with usage rights filter)")
    
    print("\n" + "="*60)
    print("⚠️  ALTERNATIVE: Use Existing Photos")
    print("="*60)
    print("\nIf you have existing photos:")
    print("  1. Copy Maliha_photos → enrolled_persons/Maliha_photos")
    print("  2. Copy Reshad_photos → enrolled_persons/Reshad_photos")
    print("  3. Add more test photos to each folder")
    print("  4. Collect unknown person photos from other sources")
    
    # Create sample download script for automated approach
    script_content = """
# Sample wget commands (if using automated download)
# Replace URLs with actual image sources

# wget -O enrolled_persons/Person1_photos/001.jpg "URL_HERE"
# wget -O enrolled_persons/Person1_photos/002.jpg "URL_HERE"
# ... etc
    """
    
    with open(base_dir / "download_commands.txt", "w") as f:
        f.write(script_content)
    
    print("\n✓ Instructions saved to: download_commands.txt")
    print("\n🚀 Next Steps:")
    print("  1. Download/copy photos to the folders")
    print("  2. Ensure at least 20-30 images per enrolled person")
    print("  3. Run the evaluation script")

if __name__ == "__main__":
    download_test_images()
