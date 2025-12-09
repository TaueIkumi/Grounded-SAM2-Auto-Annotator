import torch
import numpy as np
from typing import List, Dict, Any
from torchvision.ops import box_convert, nms
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict


class GroundedSAM2Predictor:
    def __init__(self, sam2_model_config: str, sam2_checkpoint: str,
                 grounding_dino_config: str, grounding_dino_checkpoint: str,
                 device: str = "cuda", box_threshold: float = 0.35,
                 text_threshold: float = 0.25):
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        # build SAM2 image predictor
        print("Loading SAM2...")
        sam2_model = build_sam2(sam2_model_config, sam2_checkpoint, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)

        # build grounding dino model
        print("Loading Grounding DINO...")
        self.grounding_model = load_model(
            model_config_path=grounding_dino_config, 
            model_checkpoint_path=grounding_dino_checkpoint,
            device=self.device
        )

    def predict(self, image_path: str, classes: List[str], batch_size: int = 1, multimask_output: bool = False) -> Dict[str, Any]:
        image_source, image = load_image(image_path)
        if image_source is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        self.sam2_predictor.set_image(image_source)

        all_boxes = []
        all_confidences = []
        all_labels = []

        for batch_classes in self._chunk_list(classes, batch_size):
            batch_prompt = ". ".join(batch_classes) + "."
            
            boxes, confidences, labels = predict(
                model=self.grounding_model,
                image=image,
                caption=batch_prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device
            )

            if boxes.numel() > 0:
                all_boxes.append(boxes)
                all_confidences.append(confidences)
                all_labels.extend(labels)

        if not all_boxes:
            return {
                "boxes": [], 
                "masks": [], 
                "scores": [], 
                "labels": [],
                "image_path": image_path,
                "image_shape": image_source.shape[:2]
            }

        boxes = torch.cat(all_boxes, dim=0)
        confidences = torch.cat(all_confidences, dim=0)
        
        boxes_xyxy, confidences, final_labels = self._apply_nms(boxes, confidences, all_labels, image_source.shape)

        masks, scores = self._run_sam2(boxes_xyxy, multimask_output)

        return {
            "boxes": boxes_xyxy,
            "masks": masks,
            "scores": scores,
            "labels": final_labels,
            "image_shape": image_source.shape[:2],
            "image_path": image_path
        }

    def _chunk_list(self, lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def _apply_nms(self, boxes, confidences, labels, image_shape, iou_threshold=0.5):
        h_img, w_img, _ = image_shape
        
        boxes_scaled = boxes * torch.Tensor([w_img, h_img, w_img, h_img]).to(boxes.device)
        boxes_xyxy = box_convert(boxes=boxes_scaled, in_fmt="cxcywh", out_fmt="xyxy")

        keep_indices = nms(boxes_xyxy, confidences, iou_threshold)

        boxes_xyxy = boxes_xyxy[keep_indices]
        confidences = confidences[keep_indices]
        filtered_labels = [labels[i] for i in keep_indices]

        return boxes_xyxy.cpu().numpy(), confidences.cpu().numpy(), filtered_labels

    def _run_sam2(self, input_boxes, multimask_output):
        masks, scores, _ = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=multimask_output,
        )

        if multimask_output:
            best = np.argmax(scores, axis=1)
            masks = masks[np.arange(masks.shape[0]), best]
            scores = scores[np.arange(scores.shape[0]), best]

        if masks.ndim == 4:
            masks = masks.squeeze(1)
            
        return masks, scores
    
