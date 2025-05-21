# utils/detectron_mask_generator.py

import os
import cv2
import torch
import numpy as np
from PIL import Image
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog

def setup_detectron2():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return DefaultPredictor(cfg)

def generate_mask_from_segmentation(image_path, predictor, image_size=256):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    outputs = predictor(image)

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for instance_mask in outputs["instances"].pred_masks:
        instance_mask = instance_mask.cpu().numpy().astype(np.uint8)
        mask = np.logical_or(mask, instance_mask)

    # Resize to match input size
    mask = cv2.resize(mask.astype(np.uint8) * 255, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    return Image.fromarray(mask)

# 예시 사용법
if __name__ == "__main__":
    predictor = setup_detectron2()
    img_path = "example.jpg"
    mask_img = generate_mask_from_segmentation(img_path, predictor)
    mask_img.save("seg_mask.png")
