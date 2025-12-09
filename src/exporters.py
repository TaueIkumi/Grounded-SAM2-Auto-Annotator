import cv2
import numpy as np
from typing import Dict, Any
from pathlib import Path

class BaseExporter:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

class PascalVOCSegExporter(BaseExporter):
    def save(self, result: Dict[str, Any], class_id_map: Dict[str, int], colormap: list):
        img_path = Path(result["image_path"])
        png_path = self.output_dir / f"{img_path.stem}.png"
        
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
        print(f"[PascalSeg] Saved Color Mask PNG with Void Border to {png_path}")