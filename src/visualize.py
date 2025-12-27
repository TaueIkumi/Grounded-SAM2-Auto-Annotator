import cv2
import argparse
from pathlib import Path
import supervision
from typing import Optional

class Visulizer:
    def __init__(self, datasetdir: str, output_dir: str):
        self.datasetdir = Path(datasetdir)
        if not self.datasetdir.exists():
            raise ValueError(f"Dataset directory {datasetdir} does not exist.")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_images_dir: Optional[Path] = None
    
    def load_dataset(self, format: str, task) -> supervision.DetectionDataset:
        print(f"Loading dataset in {format} format...")
        try:
            if format.lower() == "coco":
                self.current_images_dir = self.datasetdir / "images"
                ds = supervision.DetectionDataset.from_coco(
                    images_directory_path=str(self.current_images_dir),
                    annotations_path = str(self.datasetdir / "annotations" / "instances.json")
                )
            elif format.lower() == "pascal_voc":
                self.current_images_dir = self.datasetdir / "JPEGImages"
                ds = supervision.DetectionDataset.from_pascal_voc(
                    images_directory_path=str(self.current_images_dir),
                    annotations_directory_path=str(self.datasetdir / "Annotations"),
                    force_masks= (task == "seg")
                )
            else:
                raise ValueError(f"Unknown format: {format}")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise e
        
        print(f"Loaded dataset with {len(ds)} images and {len(ds.classes)} classes.")
        return ds
    
    def _read_image(self, image_name: str) -> tuple[Optional[cv2.Mat], str]:
        candidates = []
        
        if self.current_images_dir:
            candidates.append(self.current_images_dir / image_name)
            candidates.append(self.current_images_dir / Path(image_name).name)

        candidates.append(Path(image_name))
        
        candidates.append(self.datasetdir / image_name)

        print(f"Trying to load image for key '{image_name}' from paths: {[str(path) for path in candidates]}")
        for path in candidates:
            if path.exists():
                
                img = cv2.imread(str(path))
                if img is not None:
                    return img, str(path)
        
        default_path = str(candidates[0]) if candidates else image_name
        return None, default_path

    def visualize_bbox(self, dataset: supervision.DetectionDataset):
        box_annotator = supervision.BoxAnnotator()
        label_annotator = supervision.LabelAnnotator()

        image_names = list(dataset.annotations.keys())
        if not image_names:
            print("No images found in the dataset for visualization.")
            return
        
        print(f"Processing {len(image_names)} images...")

        for image_name in image_names:
            image, loaded_path = self._read_image(image_name)

            if image is None:
                print(f"Warning: Could not load image for key '{image_name}'. Checked paths like: {loaded_path}. Skipping.")
                continue

            detections = dataset.annotations[image_name]
            labels = [
                f"{dataset.classes[class_id]}"
                for class_id in detections.class_id
            ]

            annotated_frame = box_annotator.annotate(
                scene=image.copy(),
                detections=detections
            )
            
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=labels
            )

            save_name = Path(image_name).name
            cv2.imwrite(
                str(self.output_dir / save_name),
                annotated_frame
            )
        print(f"saved bounding box images to {self.output_dir}")

    def visualize_mask(self, dataset: supervision.DetectionDataset):
        mask_annotator = supervision.MaskAnnotator()

        image_names = list(dataset.annotations.keys())
        if not image_names:
            print("No images found in the dataset for visualization.")
            return
            
        print(f"Processing {len(image_names)} images for masks...")

        for image_name in image_names:
            image, loaded_path = self._read_image(image_name)

            if image is None:
                print(f"Warning: Could not load image for key '{image_name}'. Checked paths like: {loaded_path}. Skipping.")
                continue

            detections = dataset.annotations[image_name]

            # labels = [
            #     f"{dataset.classes[class_id]}"
            #     for class_id in detections.class_id
            # ]

            annotated_frame = mask_annotator.annotate(
                scene=image.copy(),
                detections=detections,
            )
            save_name = Path(image_name).name
            cv2.imwrite(
                str(self.output_dir / save_name),
                annotated_frame
            )
        print(f"saved segmentation masks to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--format", 
        type=str, 
        required=True, 
        choices=["coco", "pascal_voc"],
        help="format of the dataset to visualize (coco, pascal_voc)."
    )
    parser.add_argument(
        "--dataset-dir", 
        type=str, 
        required=True, 
        help="directory path of the dataset to visualize."
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        required=False,
        default="outputs/visualizations", 
        help="directory path to save the visualized images."
    )

    parser.add_argument(
        "--type", 
        type=str, 
        default="bbox", 
        choices=["bbox", "seg"],
        help="type of visualization to perform (bbox or mask)."
    )

    args = parser.parse_args()

    if not Path(args.dataset_dir).exists():
        print(f"Error: Dataset directory {args.dataset_dir} does not exist.")
        return
    else:
        print(f"Loading dataset from {args.dataset_dir}...")
    
    if not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        print(f"Created output directory at {args.output_dir}")

    visualizer = Visulizer(
        datasetdir=args.dataset_dir, 
        output_dir=args.output_dir
    )

    try:
        dataset = visualizer.load_dataset(
            format=args.format, 
            task=args.type,
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if args.type == "bbox":
        visualizer.visualize_bbox(dataset)
    elif args.type == "seg":
        visualizer.visualize_mask(dataset)


if __name__ == "__main__":
    main()