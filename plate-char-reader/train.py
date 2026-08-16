import os
from roboflow import Roboflow
from ultralytics import YOLO

def main():
    ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
    if not ROBOFLOW_API_KEY:
        ROBOFLOW_API_KEY = input("Please enter your Roboflow API Key: ")

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    workspace = rf.workspace("dummy-ws")
    project = workspace.project("iranian-plate-character-finder")
    version = project.version(3)
    dataset = version.download("yolo26")

    model = YOLO("yolo26n.pt") 

    print("Starting training on GPU...")
    results = model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=200,
        imgsz=640,
        batch=16,
        device=0,
        name="license_plate_yolo",
        plots=True,
        patience=30,
        workers=4
    )

if __name__ == '__main__':
    main()
