import argparse
from pathlib import Path
from . import exporters
from .GroundedSAM2Predictor import GroundedSAM2Predictor as Predictor

# [クラス名, [R, G, B]] の2次元配列リスト
VOC_NAME_COLOR_PAIRS = [
    ["background",  [0, 0, 0]],       # ID: 0
    ["aeroplane",   [128, 0, 0]],     # ID: 1
    ["bicycle",     [0, 128, 0]],     # ID: 2
    ["bird",        [128, 128, 0]],   # ID: 3
    ["boat",        [0, 0, 128]],     # ID: 4
    ["bottle",      [128, 0, 128]],   # ID: 5
    ["bus",         [0, 128, 128]],   # ID: 6
    ["car",         [128, 128, 128]], # ID: 7
    ["cat",         [64, 0, 0]],      # ID: 8
    ["chair",       [192, 0, 0]],     # ID: 9
    ["cow",         [64, 128, 0]],    # ID: 10
    ["diningtable", [192, 128, 0]],   # ID: 11
    ["dog",         [64, 0, 128]],    # ID: 12
    ["horse",       [192, 0, 128]],   # ID: 13
    ["motorbike",   [64, 128, 128]],  # ID: 14
    ["person",      [192, 128, 128]], # ID: 15
    ["pottedplant", [0, 64, 0]],      # ID: 16
    ["sheep",       [128, 64, 0]],    # ID: 17
    ["sofa",        [0, 192, 0]],     # ID: 18
    ["train",       [128, 192, 0]],   # ID: 19
    ["tvmonitor",   [0, 64, 128]]     # ID: 20
]
VOC_ID_MAP = {entry[0]: i for i, entry in enumerate(VOC_NAME_COLOR_PAIRS)}

VOC_CLASSES = [entry[0] for entry in VOC_NAME_COLOR_PAIRS if entry[0] != "background"]

VOC_COLORMAP = [entry[1] for entry in VOC_NAME_COLOR_PAIRS]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text-prompt', default=VOC_CLASSES)
    parser.add_argument('--img-path', default="Grounded-SAM-2/notebooks/images/truck.jpg")
    parser.add_argument('--sam2-checkpoint', default="/home/appuser/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument('--sam2-model-config', default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument('--grounding-dino-config', default="/home/appuser/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument('--grounding-dino-checkpoint', default="/home/appuser/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth")
    parser.add_argument('--box-threshold', type=float, default=0.35)
    parser.add_argument('--text-threshold', type=float, default=0.35)
    parser.add_argument('--device', default="cuda")
    parser.add_argument('--output-dir', default="outputs/grounded_sam2_local_demo")
    parser.add_argument('--dump-json-results', action="store_true")
    parser.add_argument('--multimask-output', action="store_true")
    parser.add_argument('--batch-size', type=int, default=1)

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
        classes=args.text_prompt,
        batch_size=args.batch_size,
        multimask_output=args.multimask_output
    )

    print(f"final labels: {result['labels']}")

    exporter = exporters.PascalVOCSegExporter(output_dir=str(args.output_dir))
    exporter.save(result, class_id_map=VOC_ID_MAP, colormap=VOC_COLORMAP)
if __name__ == "__main__":
    main()