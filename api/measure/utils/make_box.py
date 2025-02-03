import json
import re
import shutil
import cv2
import numpy as np

from pathlib import Path
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes



def _get_bboxes(img_file_list):
    bbox_map: dict[str, list[float]] = {}
    # Load a model
    model = YOLO('strawberry/models/best.pt')  # pretrained YOLOv8n model

    # Run batched inference on a list of images
    resultss = model(img_file_list)  # return a list of Results objects
    # results = results.cpu()

    # Process results list
    for idx, result in enumerate(results):
        boxes: Boxes = result.boxes  # Boxes object for bbox outputs
        bbox_map[img_file_list[idx]] = boxes.xyxy.tolist()[0]

    return bbox_map