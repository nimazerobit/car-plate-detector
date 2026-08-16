import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_DIR / "runs/detect/license_plate_yolo/weights/best.onnx"


def align_license_plate(plate_img: np.ndarray, target_width: int = 400) -> np.ndarray:
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    angles = []
    if lines is not None:
        for _, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            if -45 <= angle <= 45:
                angles.append(angle)

    height, width = plate_img.shape[:2]
    angle = float(np.median(angles)) if angles else 0.0
    matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1.0)
    aligned = cv2.warpAffine(
        plate_img,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return cv2.resize(aligned, (target_width, round(target_width * height / width)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect and align license plates.")
    parser.add_argument("image", type=Path, help="Path to the source image.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Model path.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "outputs",
        help="Directory for aligned plate crops.",
    )
    parser.add_argument("--show", action="store_true", help="Show each aligned plate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")

    model = YOLO(args.model)
    results = model(args.image, conf=args.conf, iou=args.iou, verbose=False)
    source_image = cv2.imread(str(args.image))
    if source_image is None:
        raise ValueError(f"OpenCV could not read: {args.image}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detections = 0
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            plate_crop = source_image[y1:y2, x1:x2]
            if plate_crop.size == 0:
                continue

            detections += 1
            confidence = float(box.conf[0])
            aligned_plate = align_license_plate(plate_crop)
            output_path = args.output_dir / f"{args.image.stem}_plate_{detections}.jpg"
            cv2.imwrite(str(output_path), aligned_plate)
            print(f"Plate {detections}: confidence={confidence:.2f}, saved={output_path}")

            if args.show:
                cv2.imshow(f"Detected plate {detections}", aligned_plate)
                cv2.waitKey(0)

    cv2.destroyAllWindows()
    if detections == 0:
        print("No plates were detected.")


if __name__ == "__main__":
    main()
