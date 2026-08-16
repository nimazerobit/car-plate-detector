from ultralytics import YOLO

model = YOLO("runs/detect/license_plate_yolo/weights/best.pt")

model.export(format="onnx")