import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent
DEFAULT_MODEL = REPOSITORY_ROOT / "runs/detect/plate_character_yolo/weights/best.pt"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "character_predictions.jpg",
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")

    model = YOLO(args.model)
    result = model.predict(args.image, conf=args.conf, verbose=False)[0]
    annotated = result.plot()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), annotated):
        raise OSError(f"Could not save output image: {args.output}")

    print(f"Annotated prediction saved to: {args.output}")
    if args.show:
        cv2.imshow("Detected plate characters", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
