import cv2
import json
import datetime
from typing import List
import numpy as np
from typing import Dict, Any
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom

class BaseExporter:
    def __init__(self, input_dir: str):
        self.input_dir = Path(input_dir)
        self.input_dir.mkdir(parents=True, exist_ok=True)    
    
class COCOExporter:
    def __init__(self, categories: List[str], output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.coco_format = {
            "info": {
                "description": "Auto-Annotated Dataset by Grounded-SAM-2",
                "date_created": datetime.datetime.now().isoformat()
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }

        self.class_map = {}
        for i, name in enumerate(categories):
            cat_id = i + 1
            self.class_map[name] = cat_id
            self.coco_format["categories"].append({
                "id": cat_id,
                "name": name,
                "supercategory": "object"
            })

        self.annotation_id = 1

    def add(self, result: Dict[str, Any], task: str = 'detection'):
        image_path = Path(result["image_path"])
        height, width = result["image_shape"]

        image_id = len(self.coco_format["images"]) + 1
        
        self.coco_format["images"].append({
            "id": image_id,
            "file_name": image_path.name,
            "height": height,
            "width": width,
            "date_captured": datetime.datetime.now().isoformat()
        })

        for label, mask in zip(result["labels"], result["masks"]):
            clean_label = label.strip()
            category_id = self.class_map.get(clean_label)
            
            if category_id is None:
                continue

            mask_uint8 = mask.astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
                
            c = max(contours, key=cv2.contourArea)
            area = float(cv2.contourArea(c))
            
            x, y, w, h = cv2.boundingRect(c)
            bbox = [float(x), float(y), float(w), float(h)]
            
            segmentation = []
            if task == 'segmentation':
                poly = c.flatten().tolist()
                if len(poly) >= 6:
                    segmentation = [poly]
                else:
                    continue
            else:
                segmentation = []
            
            self.coco_format["annotations"].append({
                "id": self.annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": segmentation,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0
            })
            
            self.annotation_id += 1

    def save(self):
        with open(self.output_path, "w") as f:
            json.dump(self.coco_format, f, indent=4)
        print(f"[COCO] Saved annotations to {self.output_path}")

class PascalVOCExporter(BaseExporter):
    def __init__(self, input_dir: str):
        super().__init__(input_dir)
        self.xml_dir = self.input_dir / "Annotations"
        self.xml_dir.mkdir(parents=True, exist_ok=True)

        self.seg_dir = self.input_dir / "SegmentationClass"
        self.seg_dir.mkdir(parents=True, exist_ok=True)

        self.imagesets_dir = self.input_dir / "ImageSets" / "Main"
        self.imagesets_dir.mkdir(parents=True, exist_ok=True)

        self.segmentation_dir = self.input_dir / "ImageSets" / "Segmentation"
        self.segmentation_dir.mkdir(parents=True, exist_ok=True)

        self.list_file = self.imagesets_dir / "default.txt"
        with open(self.list_file, "w") as f:
            pass

        self.segmentation_list_file = self.segmentation_dir / "default.txt"

        with open(self.segmentation_list_file, "w") as f:
            
            pass

        self.processed_files = set()
        self.processed_seg_files = set()

    def save(self, result: Dict[str, Any], class_id_map: Dict[str, int], colormap: list = None, task: str = "detection"):
        if task == "detection":
            self._save_xml(result, class_id_map)
        elif task == "segmentation":
            if colormap is None:
                raise ValueError("Segmentation task requires a colormap.")
            self._save_mask(result, class_id_map, colormap)
        else:
            raise ValueError(f"Unknown task: {task}")

    def _save_xml(self, result: Dict[str, Any], class_id_map: Dict[str, int]):
        img_path = Path(result["image_path"])
        xml_path = self.xml_dir / f"{img_path.stem}.xml"
        height, width = result["image_shape"]

        annotation = ET.Element('annotation')
        
        ET.SubElement(annotation, 'folder').text = 'images'
        ET.SubElement(annotation, 'filename').text = img_path.name
        
        source = ET.SubElement(annotation, 'source')
        ET.SubElement(source, 'database').text = 'Unknown'

        size = ET.SubElement(annotation, 'size')
        ET.SubElement(size, 'width').text = str(width)
        ET.SubElement(size, 'height').text = str(height)
        ET.SubElement(size, 'depth').text = '3'

        ET.SubElement(annotation, 'segmented').text = '0'

        for label, box in zip(result["labels"], result["boxes"]):
            clean_label = label.strip()
            if clean_label not in class_id_map:
                continue

            x1, y1, x2, y2 = box
            
            obj = ET.SubElement(annotation, 'object')
            ET.SubElement(obj, 'name').text = clean_label
            ET.SubElement(obj, 'pose').text = 'Unspecified'
            ET.SubElement(obj, 'truncated').text = '0'
            ET.SubElement(obj, 'difficult').text = '0'

            bndbox = ET.SubElement(obj, 'bndbox')
            ET.SubElement(bndbox, 'xmin').text = str(int(x1))
            ET.SubElement(bndbox, 'ymin').text = str(int(y1))
            ET.SubElement(bndbox, 'xmax').text = str(int(x2))
            ET.SubElement(bndbox, 'ymax').text = str(int(y2))

        xml_str = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="    ")
        with open(xml_path, "w") as f:
            f.write(xml_str)
        
        if img_path.stem not in self.processed_files:
            with open(self.list_file, "a") as f:
                f.write(f"{img_path.stem}\n")
            self.processed_files.add(img_path.stem)

    def _save_mask(self, result: Dict[str, Any], class_id_map: Dict[str, int], colormap: list):
        img_path = Path(result["image_path"])
        png_path = self.seg_dir / f"{img_path.stem}.png"
        
        height, width = result["image_shape"]
        seg_map = np.zeros((height, width, 3), dtype=np.uint8)

        void_color_rgb = [224, 224, 192] 
        void_color_bgr = void_color_rgb[::-1]
        kernel = np.ones((3, 3), np.uint8)

        for label, mask in zip(result["labels"], result["masks"]):
            clean_label = label.strip()
            class_id = class_id_map.get(clean_label, -1)
            
            if class_id > 0 and class_id < len(colormap):
                color_rgb = colormap[class_id]
                color_bgr = color_rgb[::-1]

                mask_uint8 = mask.astype(np.uint8)

                dilated_mask = cv2.dilate(mask_uint8, kernel, iterations=20)
                seg_map[dilated_mask.astype(bool)] = void_color_bgr

                seg_map[mask.astype(bool)] = color_bgr

        cv2.imwrite(str(png_path), seg_map)
        if img_path.stem not in self.processed_seg_files:
            with open(self.segmentation_list_file, "a") as f:
                f.write(f"{img_path.stem}\n")
            self.processed_seg_files.add(img_path.stem)
