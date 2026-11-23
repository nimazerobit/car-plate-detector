import os
import torch
from pathlib import Path
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO

load_dotenv()
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

def main():
    repository_root = Path(__file__).resolve().parent.parent
    dataset_dir = repository_root / "dataset" / "plate-detector"
    data_yaml = dataset_dir / "data.yaml"

    if not data_yaml.is_file():
        ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
        if not ROBOFLOW_API_KEY:
            ROBOFLOW_API_KEY = input("Please enter your Roboflow API Key: ")

        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        workspace = rf.workspace("zahra-bubwt")
        project = workspace.project("drhoseinpur")
        version = project.version(1)
        dataset = version.download(
            "yolo26", location=str(dataset_dir), overwrite=True
        )
        data_yaml = Path(dataset.location) / "data.yaml"

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Starting training on device: {device}...")

    model = YOLO("yolo26n.pt")

    results = model.train(
        data=str(data_yaml),
        project=str(repository_root / "runs/detect"),
        epochs=150,
        imgsz=800,
        batch=16,
        device=device,
        name="license_plate_yolo",
        plots=True,
        patience=40,
        workers=4,
        single_cls=True,
        cos_lr=True,
        degrees=12.0,
        close_mosaic=15,
        box=8.5,
        dfl=1.8,
        rect=True
    )

if __name__ == '__main__':
    main()
