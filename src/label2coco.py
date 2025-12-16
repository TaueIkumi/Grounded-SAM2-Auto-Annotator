import argparse
import yaml
from tqdm import tqdm
from pathlib import Path
from . import exporters
from .GroundedSAM2Predictor import GroundedSAM2Predictor as Predictor

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default=None, help="Directory path containing images")
    parser.add_argument('--sam2-checkpoint', default="Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument('--sam2-model-config', default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument('--grounding-dino-config', default="Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument('--grounding-dino-checkpoint', default="Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth")
    parser.add_argument('--box-threshold', type=float, default=0.35)
    parser.add_argument('--text-threshold', type=float, default=0.35)
    parser.add_argument('--device', default="cuda")
    parser.add_argument('--dump-json-results', action="store_true")
    parser.add_argument('--multimask-output', action="store_true")
    parser.add_argument('--batch-size', type=int, default=1)

    parser.add_argument('--coco-config', type=str, default="configs/coco.yaml")
    parser.add_argument('--segmentation', action='store_true', help="Enable segmentation output for COCO format.")
    
    return parser

def run_inference(args):
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_files = []

    if args.input_dir:
        input_path = Path(args.input_dir) / "images"
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(input_path.glob(ext))
    elif args.img_path:
        image_files.append(Path(args.img_path))
    else:
        raise ValueError("Either --input-dir or --img-path must be specified.")
    
    try:
        coco_data = yaml.safe_load(open(args.coco_config, 'r'))
        raw_names = coco_data.get('names')

        if not raw_names:
            raise ValueError("No 'names' field found in the COCO config file.")
        
        if isinstance(raw_names, dict):
            COCO_CLASSES = [raw_names[i] for i in range(len(raw_names))]
        elif isinstance(raw_names, list):
            COCO_CLASSES = raw_names
        else:
            raise ValueError("'names' field must be a list or a dict.")
        
        print(f"Loaded {len(COCO_CLASSES)} classes from {args.coco_config}")
        
    except Exception as e:
        raise ValueError(f"Error loading COCO config file: {e}")

    predictor = Predictor(
        sam2_model_config=args.sam2_model_config,
        sam2_checkpoint=args.sam2_checkpoint,
        grounding_dino_config=args.grounding_dino_config,
        grounding_dino_checkpoint=args.grounding_dino_checkpoint,
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold
    )


    json_path = args.output_dir / "instances.json"

    exporter = exporters.COCOExporter(
        categories=COCO_CLASSES,
        output_path=str(json_path)
    )

    print("Processing images...")

    for img_file in tqdm(image_files):
        try:
            result = predictor.predict(
                image_path=str(img_file),
                classes=COCO_CLASSES,
                batch_size=args.batch_size,
                multimask_output=args.multimask_output
            )

            exporter.add(result, task='detection')
            if args.segmentation:
                exporter.add(result, task='segmentation')
            
        except Exception as e:
            print(f"Error processing {img_file.name}: {e}")
            continue

    exporter.save()
    print(f"COCO annotations saved to {json_path}")
    return json_path

def main():
    parser = get_parser()
    args = parser.parse_args()
    run_inference(args)

if __name__ == "__main__":
    main()