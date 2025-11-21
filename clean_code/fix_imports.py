"""
Import Path Fixer for Clean Code Structure

This script updates all import statements to reflect the new clean_code structure.
"""

import os
import re
from pathlib import Path

# Define import path mappings
IMPORT_MAPPINGS = {
    # Agent imports
    r'from agent\.': 'from core.',
    r'from \.': 'from core.',
    
    # Object detection imports
    r'from BVP_LAYER01\.object_detection_with_distance_and_angle_mapping': 'from core.modules.object_detection',
    r'from object_detection_with_distance_and_angle_mapping': 'from core.modules.object_detection',
    
    # VLM imports
    r'from BVP_LAYER01\.pixtral_mistral_integration': 'from core.modules.vlm',
    r'from pixtral_mistral_integration': 'from core.modules.vlm',
    
    # Audio imports
    r'from BVP_LAYER01\.audio_transcription_whisper': 'from core.modules.audio_input',
    r'from audio_transcription_whisper': 'from core.modules.audio_input',
    r'from BVP_LAYER01\.audio_output': 'from core.modules.audio_output',
    r'from audio_output': 'from core.modules.audio_output',
    
    # Face recognition
    r'from BVP_LAYER01\.Facenet': 'from core.modules.face_recognition',
    r'from Facenet': 'from core.modules.face_recognition',
    
    # Experimental metrics
    r'from experimental_metrics': 'from core.metrics',
    
    # Adapter renames
    r'from core\.object_detector_adapter': 'from core.adapters.object_detector',
    r'from core\.scene_analysis_adapter': 'from core.adapters.scene_analysis',
    r'from core\.pixtral_analysis_adapter': 'from core.adapters.pixtral_analysis',
    r'from core\.document_scan_adapter': 'from core.adapters.document_scan',
    r'from core\.face_recognition_adapter': 'from core.adapters.face_recognition',
    r'from core\.stt_adapter': 'from core.adapters.stt',
    r'from core\.tts_adapter': 'from core.adapters.tts',
    r'from core\.chatbot_adapter': 'from core.adapters.chatbot',
    
    # Infrastructure imports
    r'from core\.tool_registry': 'from core.infrastructure.tool_registry',
    r'from core\.frame_provider': 'from core.infrastructure.frame_provider',
    r'from core\.recorder_fsm': 'from core.infrastructure.recorder_fsm',
    r'from core\.device_manager': 'from core.infrastructure.device_manager',
    r'from core\.schemas': 'from core.infrastructure.schemas',
    r'from core\.location_formatter': 'from core.infrastructure.location_formatter',
    r'from core\.unified_logger': 'from core.infrastructure.unified_logger',
    r'from core\.performance_monitor': 'from core.infrastructure.performance_monitor',
    r'from core\.observability': 'from core.infrastructure.observability',
    r'from core\.visualization': 'from core.infrastructure.visualization',
    
    # Import statements
    r'import agent\.': 'import core.',
}


def fix_imports_in_file(file_path):
    """Fix import statements in a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply all import mappings
        for pattern, replacement in IMPORT_MAPPINGS.items():
            content = re.sub(pattern, replacement, content)
        
        # Only write if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ Fixed: {file_path}")
            return True
        
        return False
        
    except Exception as e:
        print(f"✗ Error fixing {file_path}: {e}")
        return False


def main():
    """Main function to fix all imports"""
    clean_code_dir = Path(__file__).parent
    
    print("="*70)
    print("Import Path Fixer")
    print("="*70)
    print(f"Processing directory: {clean_code_dir}")
    print()
    
    # Find all Python files
    python_files = list(clean_code_dir.rglob("*.py"))
    
    fixed_count = 0
    total_count = len(python_files)
    
    for py_file in python_files:
        if py_file.name == "fix_imports.py":
            continue
        
        if fix_imports_in_file(py_file):
            fixed_count += 1
    
    print()
    print("="*70)
    print(f"Fixed {fixed_count} out of {total_count} files")
    print("="*70)


if __name__ == "__main__":
    main()
