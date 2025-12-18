import yaml
import argparse
from tqdm import tqdm
from pathlib import Path
from . import exporters
from .GroundedSAM2Predictor import GroundedSAM2Predictor as Predictor

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default=None, help="Directory path containing images")
    parser.add_argument('--sam2-checkpoint', default="/home/appuser/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument('--sam2-model-config', default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument('--grounding-dino-config', default="/home/appuser/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument('--grounding-dino-checkpoint', default="/home/appuser/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth")
    parser.add_argument('--box-threshold', type=float, default=0.5)
    parser.add_argument('--text-threshold', type=float, default=0.35)
    parser.add_argument('--device', default="cuda")
    parser.add_argument('--dump-json-results', action="store_true")
    parser.add_argument('--multimask-output', action="store_true")
    parser.add_argument('--batch-size', type=int, default=1)

    parser.add_argument('--pascal-config', type=str, default="configs/pascal_voc.yaml")
    parser.add_argument('--segmentation', action='store_true', help="Enable segmentation output for Pascal VOC format.")

    return parser
def run_inference(args):
    try:
        pascal_data = yaml.safe_load(open(args.pascal_config, 'r'))
    except Exception as e:
        raise ValueError(f"Error loading Pascal VOC config file: {e}")
    
    VOC_CLASSES = [item['name'] for item in pascal_data['colors']]
    VOC_ID_MAP = {item['name']: i+1 for i, item in enumerate(pascal_data['colors'])}
    VOC_COLORMAP = [tuple(item['rgb']) for item in pascal_data['colors']]

    INFERENCE_CLASSES = [c for c in VOC_CLASSES if c != 'background']

    print("Preparing image files...")
    print(f"classes: {VOC_CLASSES}")

    image_files = []
    if args.input_dir:
        input_path = Path(args.input_dir) / "JPEGImages"
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(input_path.glob(ext))
    elif args.img_path:
        image_files.append(Path(args.img_path))
    else:
        raise ValueError("Either --input-dir or --img-path must be specified.")
    
    predictor = Predictor(
        sam2_model_config=args.sam2_model_config,
        sam2_checkpoint=args.sam2_checkpoint,
        grounding_dino_config=args.grounding_dino_config,
        grounding_dino_checkpoint=args.grounding_dino_checkpoint,
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold
    )

    exporter = exporters.PascalVOCExporter(input_dir=str(args.input_dir))

    for img_file in tqdm(image_files):
        try:
            result = predictor.predict(
                image_path=str(img_file),
                classes=INFERENCE_CLASSES,
                batch_size=args.batch_size,
                multimask_output=args.multimask_output
            )
            exporter.save(result, class_id_map=VOC_ID_MAP, task='detection')
            
            if args.segmentation:
                exporter.save(result, class_id_map=VOC_ID_MAP, colormap=VOC_COLORMAP, task='segmentation')
                
        except Exception as e:
            print(f"\nError processing {img_file.name}: {e}")
            continue
    print(f"Pascal VOC annotations saved to {args.input_dir}")

def main():
    parser = get_parser()
    args = parser.parse_args()
    run_inference(args)

if __name__ == "__main__":
    main()