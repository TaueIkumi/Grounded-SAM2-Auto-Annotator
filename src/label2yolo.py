# label2yolo.py
from pathlib import Path
from label2coco import get_parser, run_inference
from ultralytics.data.converter import convert_coco

def main():
    parser = get_parser()
    parser.add_argument('--yolo-output-dir', default="outputs/yolo", help="Directory to save YOLO labels")
    args = parser.parse_args()

    print("=== Step 1: Generating COCO Annotations ===")

    coco_json_path = run_inference(args)
    
    print("\n=== Step 2: Converting to YOLO Format ===")
    
    yolo_out_dir = Path(args.yolo_output_dir)
    yolo_out_dir.mkdir(parents=True, exist_ok=True)
    
    use_segments = (args.coco_task == 'segmentation')
    
    try:
        labels_dir = coco_json_path.parent
        
        convert_coco(
            labels_dir=str(labels_dir),
            save_dir=str(yolo_out_dir),
            use_segments=use_segments
        )
        
        print("✅ Conversion complete!")
        print(f"   Mode: {'Segmentation' if use_segments else 'Bounding Box'}")
        print(f"   Output: {yolo_out_dir}")
        
    except Exception as e:
        print(f"❌ Error during YOLO conversion: {e}")

if __name__ == "__main__":
    main()