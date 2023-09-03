import argparse
import os
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests


yolov7_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vendor/yolov7')
sys.path.append(yolov7_path)

from models.experimental import attempt_load
import torch
from torchvision import transforms
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt, strip_optimizer, xyxy2xywh
from utils.plots import colors, output_to_keypoint, plot_one_box_kpt, plot_skeleton_kpts
from utils.torch_utils import select_device
import asyncio
import websockets
import json


HOST = 'localhost'
PORT = '8080'

async def predict_request(payload):
    """
    Args:
        - payload: {'array': (1, 10, 50) shape (10 frames)}
    Returns:
        - score: Value between 0 and 1
        - status: Good or bad posture (depending on threshold:0.7)
    """
    uri = f"ws://{HOST}:{PORT}"
    try:
        async with websockets.connect(uri) as ws:
            payload_json = json.dumps(payload)
            await ws.send(payload_json)
            raw_prediction = await ws.recv()
            prediction = json.loads(raw_prediction)
            score = prediction['score']
            status = prediction['status']
            return score, status
    except:
        return None, 'server-error'


def predict_http_request(payload):
    response = requests.post(f"http://{HOST}:{PORT}/predict", json=payload)
    if response.status_code == 200:
        data = response.json()
        print("Response:")
        print(json.dumps(data, indent=4))
    else:
        print("Error:", response.status_code)
        print(response.text)

@torch.no_grad()
def run(poseweights="yolov7-w6-pose.pt",source="football1.mp4",device='cpu',view_img=False,
        save_conf=False,line_thickness = 3,hide_labels=False, hide_conf=True):
    # global ap_model

    frame_count = 0  #count no of frames
    total_fps = 0  #count total fps
    time_list = []   #list to store time
    fps_list = []    #list to store fps
    
    device = select_device(opt.device) #select device
    model = attempt_load(poseweights, map_location=device)  #Load model
    _ = model.eval()
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
   
    if source.isnumeric() :    
        cap = cv2.VideoCapture(int(source))    #pass video to videocapture object
    else :
        cap = cv2.VideoCapture(source)    #pass video to videocapture object
   
    if (cap.isOpened() == False):   #check if videocapture not opened
        print('Error while trying to read video. Please check path again')
        raise SystemExit()

    else:
        frame_width = int(cap.get(3))  #get video frame width
        vid_write_image = letterbox(cap.read()[1], (frame_width), stride=64, auto=True)[0] #init videowriter
        resize_height, resize_width = vid_write_image.shape[:2]
        out = cv2.VideoWriter(f"{source}_keypoint.mp4",
                            cv2.VideoWriter_fourcc(*'mp4v'), 30,
                            (resize_width, resize_height))

        current_sequence = []
        while(cap.isOpened):
            ret, frame = cap.read() 
            if ret: 
                orig_image = frame 
                image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB) 
                image = letterbox(image, (frame_width), stride=64, auto=True)[0]
                image = transforms.ToTensor()(image)
                image = torch.tensor(np.array([image.numpy()]))
                image = image.to(device)
                image = image.float()
                start_time = time.time()
            
                with torch.no_grad():  #get predictions
                    output_data, _ = model(image)

                output_data = non_max_suppression_kpt(output_data,   #Apply non max suppression
                                            0.25,   # Conf. Threshold.
                                            0.65, # IoU Threshold.
                                            nc=model.yaml['nc'], # Number of classes.
                                            nkpt=model.yaml['nkpt'], # Number of keypoints.
                                            kpt_label=True)
            
                output = output_to_keypoint(output_data)

                if frame_count % 10 == 0:
                    landmarks = output[0, 7:].T
                    current_sequence += [landmarks[:-1]]

                if len(current_sequence) == 10:
                    current_sequence = np.array([current_sequence])
                    print(current_sequence.shape)
                    payload = {'array': current_sequence.tolist() }
                    predict_http_request(payload)
                    # score, status = asyncio.run(predict_request(payload))
                    # if status == 'server-error':
                    #     print('Server error or server not launched')
                    # print(score, status)
                    current_sequence = []


                im0 = image[0].permute(1, 2, 0) * 255 # Change format [b, c, h, w] to [h, w, c] for displaying the image.
                im0 = im0.cpu().numpy().astype(np.uint8)
                
                im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR) #reshape image format to (BGR)
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                for i, pose in enumerate(output_data):  # detections per image
                
                    if len(output_data):  #check if no pose
                        for c in pose[:, 5].unique(): # Print results
                            n = (pose[:, 5] == c).sum()  # detections per class
                            # print("No of Objects in Current Frame : {}".format(n))
                        
                        for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose[:,:6])): #loop over poses for drawing on frame
                            c = int(cls)  # integer class
                            kpts = pose[det_index, 6:]
                            label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                            plot_one_box_kpt(xyxy, im0, label=label, color=colors(c, True), 
                                        line_thickness=opt.line_thickness,kpt_label=True, kpts=kpts, steps=3, 
                                        orig_shape=im0.shape[:2])

                
                end_time = time.time()  #Calculatio for FPS
                fps = 1 / (end_time - start_time)
                total_fps += fps
                frame_count += 1
                
                fps_list.append(total_fps) #append FPS in list
                time_list.append(end_time - start_time) #append time in list
                
                # Stream results
                if view_img:
                    cv2.imshow("YOLOv7 Pose Estimation Demo", im0)
                    key = cv2.waitKey(1) & 0xFF  # Wait for 1 millisecond and get the pressed key
                    if key == ord('q'):
                        cv2.destroyAllWindows()  # Close the window if 'q' is pressed
                        break

                out.write(im0)  #writing the video frame

            else:
                break

        cap.release()
        # cv2.destroyAllWindows()
        avg_fps = total_fps / frame_count
        print(f"Average FPS: {avg_fps:.3f}")
        
        #plot the comparision graph
        plot_fps_time_comparision(time_list=time_list,fps_list=fps_list)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='src_models/yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='football1.mp4', help='video/0 for webcam') #video source
    parser.add_argument('--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)')   #device arugments
    parser.add_argument('--view-img', action='store_true', help='display results')  #display results
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels') #save confidence in txt writing
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)') #box linethickness
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels') #box hidelabel
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences') #boxhideconf
    opt = parser.parse_args()
    return opt

#function for plot fps and time comparision graph
def plot_fps_time_comparision(time_list,fps_list):
    plt.figure()
    plt.xlabel('Time (s)')
    plt.ylabel('FPS')
    plt.title('FPS and Time Comparision Graph')
    plt.plot(time_list, fps_list,'b',label="FPS & Time")
    plt.savefig("FPS_and_Time_Comparision_pose_estimate.png")
    

#main function
def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    strip_optimizer(opt.device,opt.poseweights)
    main(opt)
