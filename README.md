# Iranian License Plate Detector

This project uses two YOLO26 models to detect Iranian vehicle license plates and their characters:

- **Plate detector:** finds and aligns plates in vehicle images.
- **Character detector:** identifies digits and Persian letter codes on a plate crop.

Both `.pt` and `.onnx` model weights are stored under `runs/detect/`.

## Installation

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

## Usage

Detect and align plates in an image:

```powershell
python plate-detector/test.py path\to\vehicle-image.jpg
```

Detected crops are saved to `plate-detector/outputs/`. Use `--conf`, `--iou`, `--output-dir`, `--show`, or `--model` to customize detection.

Detect the characters in a plate crop:

```powershell
python plate-char-reader/test.py plate-detector\outputs\vehicle-image_plate_1.jpg
```

The annotated result is saved to `plate-char-reader/outputs/character_predictions.jpg`. Use `--output`, `--show`, `--conf`, or `--model` for alternatives.

## Training

Set your Roboflow API key when needed:

```powershell
$env:ROBOFLOW_API_KEY = "YOUR_API_KEY"
```

The training scripts download the datasets if local `data.yaml` files are absent:

```powershell
python plate-detector/train.py
python plate-char-reader/train.py
```

Training artifacts are written to `runs/detect/license_plate_yolo/` and `runs/detect/plate_character_yolo/`.

Convert trained PyTorch weights to ONNX with `plate-detector/convert_to_onnx.py` or `plate-char-reader/convert_to_onnx.py`.

## Model Results

Final validation metrics recorded in `results.csv`:

| Model | Epochs | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|---:|
| Plate detector (`yolo26n`) | 144 | 99.52% | 99.65% | 99.50% | 75.28% |
| Character detector (`yolo26s`) | 94 | 98.26% | 98.37% | 99.02% | 77.58% |

The character model supports 30 classes: the digits `0-9` and 20 Iranian letter codes.
