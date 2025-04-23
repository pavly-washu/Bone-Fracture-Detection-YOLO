import argparse
from ultralytics import YOLO
import os
import sys
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    model = YOLO("yolo11n.pt")
    results = model.train(data="data.yaml", epochs=200, imgsz=640, batch=0.6, device=DEVICE)

if __name__ == '__main__':
    main()
