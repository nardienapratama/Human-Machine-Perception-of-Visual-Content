import random

import torch
from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from torchvision.models import detection
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np
import colorsys


def getMLModel(model_type, args=None):
    model_dict = dict()
    try:
        if model_type == "object-detection":
            # set the device we will be using to run the model
            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(DEVICE)

            # load the list of categories in the COCO dataset and then generate a
            # set of bounding box colors for each class
            with open(args["labels"]) as file:
                CLASSES = [line.rstrip() for line in file]

            COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

            # initialize a dictionary containing model name and its corresponding 
            # torchvision function call
            MODELS = {
                "frcnn-resnet": detection.fasterrcnn_resnet50_fpn_v2,
                "frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_320_fpn,
                "retinanet": detection.retinanet_resnet50_fpn
            }

            WEIGHTS = {
                "frcnn-resnet": detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
                "frcnn-mobilenet": detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1,
                "retinanet": detection.RetinaNet_ResNet50_FPN_Weights.COCO_V1
            }

            # load the model and set it to evaluation mode
            model = MODELS[args["model"]](progress=True,
                num_classes=len(CLASSES), weights=WEIGHTS[args["model"]]).to(DEVICE)
            # model.eval()
            model_dict["model"] = model
            model_dict["weights"] = WEIGHTS[args["model"]]
            if not args:
                raise ValueError
            model_dict["args"] = args
            model_dict["device"] = DEVICE

        elif model_type == "image-captioning":
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            model_dict["model"] = model
            model_dict["processor"] = processor

        else:
            raise ValueError("Error: No model is returned.")
        return model_dict

    except ValueError as ve:
        print(ve)

def generate_random_color():
    hue = random.uniform(0, 1)
    saturation = random.uniform(0.5, 1)  # Adjust this range for saturation
    value = random.uniform(0.8, 1)  # Adjust this range for brightness

    # Convert HSV to RGB
    rgb_color = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue, saturation, value))

    return rgb_color