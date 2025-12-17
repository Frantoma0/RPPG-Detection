#!/usr/bin/env python3
"""
Quick launcher for rPPG Lie Detection System v3.0

Usage:
    python run.py           # Launches Streamlit version (recommended)
    python run.py --legacy  # Launches Flask version (old)
"""

import sys
import subprocess
import os

def check_dependencies():
    """Check if required packages are installed"""
    required = ['streamlit', 'cv2', 'numpy', 'scipy']
    missing = []

    for pkg in required:
        try:
            if pkg == 'cv2':
                import cv2
            else:
                __import__(pkg)
        except ImportError:
            missing.append(pkg if pkg != 'cv2' else 'opencv-python')

    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    return True

def check_gpu():
    """Check GPU availability"""
    # CuPy
    try:
        import cupy as cp
        if cp.cuda.is_available():
            props = cp.cuda.runtime.getDeviceProperties(0)
            print(f"[GPU] CuPy CUDA: {props['name'].decode()}")
            return True
    except:
        pass

    # PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[GPU] PyTorch CUDA: {torch.cuda.get_device_name(0)}")
            return True
    except:
        pass

    print("[GPU] No CUDA acceleration available - running on CPU")
    return False

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║           rPPG LIE DETECTION SYSTEM v3.0 - LAUNCHER                  ║
    ║           Optimized for NVIDIA RTX 4070 Laptop GPU                   ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Check GPU
    check_gpu()
    print()

    # Determine which version to run
    if '--legacy' in sys.argv:
        print("Starting Flask version (legacy)...")
        print("Open browser: http://localhost:5000")
        subprocess.run([sys.executable, 'rppg_lie_detector.py'])
    else:
        print("Starting Streamlit version...")
        print("Browser will open automatically")
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run',
            'rppg_streamlit.py',
            '--server.headless', 'false',
            '--browser.gatherUsageStats', 'false',
            '--theme.base', 'dark'
        ])

if __name__ == '__main__':
    main()
