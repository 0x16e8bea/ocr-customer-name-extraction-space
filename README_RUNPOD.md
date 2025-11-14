# RunPod Deployment Guide

## Quick Start

1. **Create a RunPod GPU Pod** with this image:
   ```
   runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04
   ```

2. **SSH into the pod** and run:
   ```bash
   git clone https://github.com/0x16e8bea/ocr-customer-name-extraction-space.git
   cd ocr-customer-name-extraction-space
   pip install --upgrade torch torchvision
   pip install -r requirements.txt
   python app.py
   ```

3. **Access the app** via the RunPod HTTP port 7860 URL.

## Requirements

- **Disk Space**: 25 GB minimum (for models and dependencies)
- **GPU**: Sufficient VRAM for Qwen2-VL OCR model

## Notes

- The `spaces.py` mock module is included to replace the HuggingFace Spaces GPU decorator
- Models will be downloaded automatically on first run (may take a few minutes)
