from pathlib import Path
from ultralytics import YOLO

repository_root = Path(__file__).resolve().parent.parent
model = YOLO(repository_root / "runs/detect/license_plate_yolo/weights/best.pt")

model.export(format="onnx")