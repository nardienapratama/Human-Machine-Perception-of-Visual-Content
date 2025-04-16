# Reference: https://pyimagesearch.com/2021/08/02/pytorch-object-detection-with-pre-trained-networks/
from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

from torchvision.models import detection
import numpy as np
import argparse
import pickle
import torch
import cv2


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
	help="path to the input image")
ap.add_argument("-m", "--model", type=str, default="frcnn-resnet",
	choices=["frcnn-resnet", "frcnn-mobilenet", "retinanet"],
	help="name of the object detection model")
ap.add_argument("-l", "--labels", type=str, default="coco-labels-paper.txt",
	help="path to file containing list of categories in COCO dataset")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# set the device we will be using to run the model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(DEVICE)

# load the list of categories in the COCO dataset and then generate a
# set of bounding box colors for each class
with open(args["labels"],) as file:
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
model.eval()

import os
rootdir = '/home/nardienapratama/winter-research-2023/winter-research-2023/data/images'

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if file.endswith(".jpg"):
            print(file)
            
            file_path = os.path.join("data/images", file)

            # load the image from disk
            image = read_image(file_path)

            # Step 2: Initialize the inference transforms
            preprocess = WEIGHTS[args["model"]].transforms()

            img = image.to(DEVICE)

            # Step 3: Apply inference preprocessing transforms
            batch = [preprocess(img)]
            prediction = model(batch)[0]

            # loop over the detections
            for i in range(0, len(prediction["boxes"])):
                # extract the confidence (i.e., probability) associated with the
                # prediction
                confidence = prediction["scores"][i]
                # filter out weak detections by ensuring the confidence is
                # greater than the minimum confidence
                if confidence > args["confidence"]:
                    # extract the index of the class label from the detections,
                    # then compute the (x, y)-coordinates of the bounding box
                    # for the object
                    idx = int(prediction["labels"][i])
                    box = prediction["boxes"][i].detach().cpu().numpy()
                    (startX, startY, endX, endY) = box.astype("int")
                    # display the prediction to our terminal
                    label = "{}: {:.2f}%".format(WEIGHTS[args["model"]].meta["categories"][idx], confidence * 100)
                    print("[INFO] {}".format(label))

            labels = [WEIGHTS[args["model"]].meta["categories"][i] for i in prediction["labels"]]
            print(labels)
            box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                                    labels=labels,
                                    colors="red",
                                    width=4, font_size=30)
                                    
            # im = to_pil_image(box.detach())
            # im.show()

