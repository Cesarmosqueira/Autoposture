import sys
sys.path.append('landmark_extraction_yolo')  # Add the parent directory to the sys.path
# RUNNING THIS FROM ROOT

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import urllib.request
import sys
import torch
import time
import datetime

from torchvision import transforms
from PIL import Image

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# if not found
# https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt
def loading_yolov7_model(yolomodel):
    """
    Loading yolov7 model
    """
    print("Loading model:", yolomodel)
    model = torch.load(yolomodel, map_location=device)['model']
    model.float().eval()

    if torch.cuda.is_available():
        # half() turns predictions into float16 tensors
        # which significantly lowers inference time
        model.half().to(device)

    return model, yolomodel


try:
    print("Loading the model...")
    model, yolomodel = loading_yolov7_model(yolomodel='models/yolov7-w6-pose.pt')
    print("Using the", 'yolov7-w6-pose.pt', "model")
    print("Done")

except:
    print("[Error] Cannot load the model", 'yolov7-w6-pose.pt')


def running_inference(image):
    """
    Running yolov7 model inference
    """
    image = letterbox(image, 960,
                      stride=64,
                      auto=True)[0]  # shape: (567, 960, 3)
    image = transforms.ToTensor()(image)  # torch.Size([3, 567, 960])

    if torch.cuda.is_available():
        image = image.half().to(device)

    image = image.unsqueeze(0)  # torch.Size([1, 3, 567, 960])

    with torch.no_grad():
        output, _ = model(image)

    return output, image

def draw_keypoints(output, image, confidence=0.25, threshold=0.65):
    """
    Draw YoloV7 pose keypoints
    """
    output = non_max_suppression_kpt(
        output,
        confidence,  # Confidence Threshold
        threshold,  # IoU Threshold
        nc=model.yaml['nc'],  # Number of Classes
        nkpt=model.yaml['nkpt'],  # Number of Keypoints
        kpt_label=True)

    with torch.no_grad():
        keypoints = output_to_keypoint(output)

    nimg = image[0].permute(1, 2, 0) * 255
    nimg = cv2.cvtColor(nimg.cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)

    for idx in range(keypoints.shape[0]):
        joints = keypoints[idx, 7:].T
        joint_number = 12
        joints[joint_number*3:joint_number*3 + 3] = 0
        plot_skeleton_kpts(nimg, joints, 3)

    return keypoints, nimg



