import argparse
from pathlib import Path
from . import exporters
from .GroundedSAM2Predictor import GroundedSAM2Predictor as Predictor

COCO_CLASSES = [
    "person",          # 0
    "bicycle",         # 1
    "car",             # 2
    "motorcycle",      # 3
    "airplane",        # 4
    "bus",             # 5
    "train",           # 6
    "truck",           # 7
    "boat",            # 8
    "traffic light",   # 9
    "fire hydrant",    # 10
    "stop sign",       # 11
    "parking meter",   # 12
    "bench",           # 13
    "bird",            # 14
    "cat",             # 15
    "dog",             # 16
    "horse",           # 17
    "sheep",           # 18
    "cow",             # 19
    "elephant",        # 20
    "bear",            # 21
    "zebra",           # 22
    "giraffe",         # 23
    "backpack",        # 24
    "umbrella",        # 25
    "handbag",         # 26
    "tie",             # 27
    "suitcase",        # 28
    "frisbee",         # 29
    "skis",            # 30
    "snowboard",       # 31
    "sports ball",     # 32
    "kite",            # 33
    "baseball bat",    # 34
    "baseball glove",  # 35
    "skateboard",      # 36
    "surfboard",       # 37
    "tennis racket",   # 38
    "bottle",          # 39
    "wine glass",      # 40
    "cup",             # 41
    "fork",            # 42
    "knife",           # 43
    "spoon",           # 44
    "bowl",            # 45
    "banana",          # 46
    "apple",           # 47
    "sandwich",        # 48
    "orange",          # 49
    "broccoli",        # 50
    "carrot",          # 51
    "hot dog",         # 52
    "pizza",           # 53
    "donut",           # 54
    "cake",            # 55
    "chair",           # 56
    "couch",           # 57
    "potted plant",    # 58
    "bed",             # 59
    "dining table",    # 60
    "toilet",          # 61
    "tv",              # 62
    "laptop",          # 63
    "mouse",           # 64
    "remote",          # 65
    "keyboard",        # 66
    "cell phone",      # 67
    "microwave",       # 68
    "oven",            # 69
    "toaster",         # 70
    "sink",            # 71
    "refrigerator",    # 72
    "book",            # 73
    "clock",           # 74
    "vase",            # 75
    "scissors",        # 76
    "teddy bear",      # 77
    "hair drier",      # 78
    "toothbrush"       # 79
]

COCO_ID_MAP = {name: i for i, name in enumerate(COCO_CLASSES)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path', default="Grounded-SAM-2/notebooks/images/truck.jpg")
    parser.add_argument('--sam2-checkpoint', default="/home/appuser/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument('--sam2-model-config', default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument('--grounding-dino-config', default="/home/appuser/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument('--grounding-dino-checkpoint', default="/home/appuser/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth")
    parser.add_argument('--box-threshold', type=float, default=0.35)
    parser.add_argument('--text-threshold', type=float, default=0.35)
    parser.add_argument('--device', default="cuda")
    parser.add_argument('--output-dir', default="outputs/coco")
    parser.add_argument('--dump-json-results', action="store_true")
    parser.add_argument('--multimask-output', action="store_true")
    parser.add_argument('--batch-size', type=int, default=1)

    parser.add_argument('--coco-task', choices=['detection', 'segmentation'], default='detection')
    args = parser.parse_args()

    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictor = Predictor(
        sam2_model_config=args.sam2_model_config,
        sam2_checkpoint=args.sam2_checkpoint,
        grounding_dino_config=args.grounding_dino_config,
        grounding_dino_checkpoint=args.grounding_dino_checkpoint,
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold
    )

    result = predictor.predict(
        image_path=args.img_path,
        classes=COCO_CLASSES,
        batch_size=args.batch_size,
        multimask_output=args.multimask_output
    )

    json_path = args.output_dir / "annotations.json"
    print(f"final labels: {result['labels']}")

    exporter = exporters.COCOExporter(
        categories=COCO_CLASSES,
        output_path=str(json_path)
    )

    exporter.add(result, task=args.coco_task)
    exporter.save()
if __name__ == "__main__":
    main()