"""
kinect_det conda 환경 검증 스크립트
실행: conda activate kinect_det && python verify_env.py
"""
import sys
print(f"Python: {sys.version.split()[0]}")

import torch
print(f"torch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")

import torchvision, torchaudio
print(f"torchvision: {torchvision.__version__}")
print(f"torchaudio: {torchaudio.__version__}")

import cv2; print(f"cv2: {cv2.__version__}")
import numpy; print(f"numpy: {numpy.__version__}")
import ultralytics; print(f"ultralytics: {ultralytics.__version__}")
import supervision; print(f"supervision: {supervision.__version__}")
import transformers; print(f"transformers: {transformers.__version__}")
import timm; print(f"timm: {timm.__version__}")
import open3d; print(f"open3d: {open3d.__version__}")
import onnxruntime; print(f"onnxruntime: {onnxruntime.__version__}")
import einops; print("einops: OK")
import omegaconf; print(f"omegaconf: {omegaconf.__version__}")
import shapely; print(f"shapely: {shapely.__version__}")
import sahi; print(f"sahi: {sahi.__version__}")

try:
    import groundingdino; print("groundingdino: OK")
except Exception as e:
    print(f"groundingdino: {e}")

try:
    import pyk4a; print(f"pyk4a: {pyk4a.__version__}")
except Exception as e:
    print(f"pyk4a: {e}")

print("=== All checks done ===")
