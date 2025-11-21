#!/usr/bin/env python3
"""
Test script for cross-platform functionality.

This script tests GPU/MPS detection and cross-platform keypress handling.
"""

import sys
import torch
from utils.cross_platform_keypress import wait_for_r_key, CrossPlatformKeypress


def test_gpu_detection():
    """Test GPU/MPS detection functionality."""
    print("Testing GPU/MPS Detection")
    print("=" * 40)
    
    # Test CUDA
    if torch.cuda.is_available():
        print(f"CUDA: Available (GPU: {torch.cuda.get_device_name(0)})")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("CUDA: Not available")
    
    # Test MPS
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("MPS: Available (Apple Silicon GPU)")
    else:
        print("MPS: Not available")
    
    # Test optimal device selection
    if torch.cuda.is_available():
        optimal_device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        optimal_device = 'mps'
    else:
        optimal_device = 'cpu'
    
    print(f"Optimal device: {optimal_device.upper()}")
    print()


def test_keypress_functionality():
    """Test cross-platform keypress functionality."""
    print("Testing Cross-Platform Keypress")
    print("=" * 40)
    
    try:
        handler = CrossPlatformKeypress()
        print(f"Keypress handler initialized for platform: {sys.platform}")
        
        print("\nTesting keypress detection...")
        print("Press 'r' to test keypress detection (or 'q' to quit):")
        
        if wait_for_r_key("Press 'r' to test:", timeout=10):
            print("'r' key detected successfully!")
        else:
            print("Timeout waiting for 'r' key")
        
        print("\nPress 'q' to quit the test:")
        if wait_for_r_key("Press 'q' to quit:", timeout=10):
            print("Test completed!")
        
    except Exception as e:
        print(f"Keypress test failed: {e}")
        print("   This might be expected in some environments (e.g., Jupyter notebooks)")


def test_device_usage():
    """Test device usage in PyTorch."""
    print("Testing Device Usage")
    print("=" * 40)
    
    try:
        # Test tensor creation on optimal device
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        
        print(f"Using device: {device}")
        
        # Create a test tensor
        x = torch.randn(3, 3).to(device)
        print(f"Created tensor on {device}: {x.device}")
        
        # Test basic operations
        y = x @ x.T
        print(f"Matrix multiplication successful: {y.shape}")
        
        # Test memory usage (CUDA only)
        if device == 'cuda':
            memory_allocated = torch.cuda.memory_allocated() / 1024**2
            memory_reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"GPU Memory - Allocated: {memory_allocated:.1f} MB, Reserved: {memory_reserved:.1f} MB")
        
    except Exception as e:
        print(f"Device test failed: {e}")


def main():
    """Run all tests."""
    print("Blind Vision - Cross-Platform Functionality Test")
    print("=" * 60)
    print(f"Platform: {sys.platform}")
    print(f"Python: {sys.version}")
    print()
    
    test_gpu_detection()
    test_device_usage()
    
    print("\n" + "=" * 60)
    print("Keypress test requires interactive input.")
    print("Run this script in a terminal to test keypress functionality.")
    print("=" * 60)
    
    # Only run keypress test if running in terminal
    if sys.stdin.isatty():
        test_keypress_functionality()
    else:
        print("Skipping keypress test (not running in terminal)")


if __name__ == "__main__":
    main()
