# Iranian License Plate Detection

<img width="2048" height="620" alt="sample" src="https://github.com/user-attachments/assets/40414b7e-fbe1-49ff-8b0c-8595fe63ef3c" />

This repository contains two Ultralytics YOLO models for processing Iranian vehicle license plates:

1. **Plate detector** — detects a license plate in a vehicle image and saves an aligned crop.
2. **Character detector** — detects the characters in a cropped plate image and saves an annotated result.

Pretrained model weights are included in both PyTorch (`.pt`) and ONNX (`.onnx`) formats, together with training metrics and visualizations.

## Project structure

```text
car-plate-detector/
├── plate-detector/
│   ├── train.py
│   ├── test.py
│   ├── convert_to_onnx.py
│   └── runs/detect/license_plate_yolo/weights/
│       ├── best.pt
│       └── best.onnx
├── plate-char-reader/
│   ├── train.py
│   ├── test.py
│   ├── convert_to_onnx.py
│   └── runs/detect/license_plate_yolo/weights/
│       ├── best.pt
│       └── best.onnx
└── requirements.txt
```

## Requirements

- Python 3.10 or newer
- NVIDIA GPU and CUDA are recommended for training; the current training scripts use `device=0`.
- A Roboflow account and API key are required only to download datasets and train models.

## Installation

Create and activate a virtual environment from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the plate detector

Pass any vehicle image to the detector. The script uses the included ONNX model by default and saves every aligned plate crop to `plate-detector/outputs/`.

```powershell
python plate-detector/test.py path\to\vehicle-image.jpg
```

Optional flags:

```powershell
python plate-detector/test.py path\to\vehicle-image.jpg --conf 0.35 --show
```

- `--conf`: minimum confidence score (default: `0.25`)
- `--iou`: IoU threshold for non-maximum suppression (default: `0.45`)
- `--output-dir`: directory for aligned plate crops
- `--show`: display each detected and aligned plate
- `--model`: use a custom `.pt` or `.onnx` model path

## Run the character detector

Give the character detector a cropped plate image. It uses the included PyTorch model by default and saves an annotated image to `plate-char-reader/outputs/character_predictions.jpg`.

```powershell
python plate-char-reader/test.py path\to\cropped-plate.jpg
```

To save somewhere else or preview the result:

```powershell
python plate-char-reader/test.py path\to\cropped-plate.jpg --output output.jpg --show
```

## Recommended workflow

Run the plate detector first, then provide one of its output crops to the character detector:

```powershell
python plate-detector/test.py path\to\vehicle-image.jpg
python plate-char-reader/test.py plate-detector\outputs\vehicle-image_plate_1.jpg
```

## Training

Set your Roboflow API key:

```powershell
$env:ROBOFLOW_API_KEY = "YOUR_API_KEY"
```

Train the plate detector:

```powershell
cd plate-detector
python train.py
```

Train the character detector:

```powershell
cd plate-char-reader
python train.py
```

Each script downloads its configured Roboflow dataset and trains `yolo26n.pt` for up to 200 epochs at an image size of 640 and batch size of 16. Training artifacts are saved under `runs/detect/license_plate_yolo/`.

> If `ROBOFLOW_API_KEY` is not set, the training scripts prompt for it interactively. Never commit this key.

## Export to ONNX

Convert the best PyTorch weights to ONNX:

```powershell
cd plate-detector
python convert_to_onnx.py
```

Or:

```powershell
cd plate-char-reader
python convert_to_onnx.py
```

## Training results

The following values are from the final recorded epoch in each `results.csv` file:

| Model | Precision | Recall | mAP@50 | mAP@50-95 |
| --- | ---: | ---: | ---: | ---: |
| Plate detector | 99.20% | 97.90% | 99.38% | 74.68% |
| Character detector | 97.40% | 98.15% | 98.91% | 79.40% |

Learning curves, confusion matrices, prediction samples, and full metrics are available in each model's `runs/detect/license_plate_yolo/` directory.

