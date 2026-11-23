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
    dataset_dir = repository_root / "dataset" / "plate-char-reader"
    data_yaml = dataset_dir / "data.yaml"

    if not data_yaml.is_file():
        ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
        if not ROBOFLOW_API_KEY:
            ROBOFLOW_API_KEY = input("Please enter your Roboflow API Key: ")

        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        workspace = rf.workspace("dummy-ws")
        project = workspace.project("iranian-plate-character-finder")
        version = project.version(int(3))
        dataset = version.download(
            "yolo26", location=str(dataset_dir), overwrite=True
        )
        data_yaml = Path(dataset.location) / "data.yaml"

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Starting character detector training on device: {device}...")

    model = YOLO("yolo26s.pt")

    results = model.train(
        data=str(data_yaml),
        project=str(repository_root / "runs/detect"),
        epochs=160,
        imgsz=416,
        batch=-1,
        device=device,
        name="plate_character_yolo",
        plots=True,
        patience=30,
        workers=4,
        rect=False,
        cos_lr=True,
        
        # OCR essential augmentations
        fliplr=0.0,
        flipud=0.0,
        mosaic=0.0,
        erasing=0.0,
        
        # Night / low light / low quality camera augmentations
        degrees=6.0,
        scale=0.3,
        
        # Loss balance for 30 fine-grained character classes
        cls=1.1,
        box=7.5
    )

if __name__ == '__main__':
    main()
