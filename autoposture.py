import argparse
import os
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
from internal.prediction_client import predict_http_request


yolov7_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vendor/yolov7')
sys.path.append(yolov7_path)

from models.experimental import attempt_load
import torch
from torchvision import transforms
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt, strip_optimizer, xyxy2xywh
from utils.plots import colors, output_to_keypoint, plot_one_box_kpt, plot_skeleton_kpts
from utils.torch_utils import select_device
# from tts.tttest import generate_audios, play_audio
import asyncio
import threading
import websockets
import json


POSEWEIGHTS = 'src_models/yolov7-w6-pose.pt'


@torch.no_grad()
def run(source, device, separation, length, multiple):
    # global ap_model
    separation = int(separation)
    length = int(length)

    frame_count = 0  #count no of frames
    total_fps = 0  #count total fps
    
    device = select_device(opt.device) #select device
    model = attempt_load(POSEWEIGHTS, map_location=device)  #Load model
    _ = model.eval()
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
   
    if source.isnumeric() :    
        cap = cv2.VideoCapture(int(source))    #pass video to videocapture object
    else:
        cap = cv2.VideoCapture(source)    #pass video to videocapture object
   
    if (cap.isOpened() == False):   #check if videocapture not opened
        print('Error while trying to read video. Please check path again')
        raise SystemExit()

    else:
        frame_width = int(cap.get(3))  #get video frame width
        # logic for multiple persons
        people = {}
        next_object_id = 0
        # logic for single persons
        current_sequence = []
        current_score = 0
        current_status = 'good'
        previous_status = "None"
        longevity = 0 # frames spent in the current status

        # generate_audios("good"); generate_audios("bad")
        # bad_audio_thread = threading.Thread(target=play_audio, args=["bad"])

        empty = False
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
            
                with torch.no_grad():  #get predictions
                    output_data, _ = model(image)

                output_data = non_max_suppression_kpt(output_data,   #Apply non max suppression
                                            0.25,   # Conf. Threshold.
                                            0.65, # IoU Threshold.
                                            nc=model.yaml['nc'], # Number of classes.
                                            nkpt=model.yaml['nkpt'], # Number of keypoints.
                                            kpt_label=True)
            
                output = output_to_keypoint(output_data)
                if multiple:
                    if len(output) == 0:
                        if not empty:
                            print("Wiping data, waiting for objects to appear in frame")
                        people = {}
                        next_object_id = 0
                        empty = True
                    else:
                        empty = False
                else:
                    if output.shape[0] > 0:
                        if frame_count % separation == 0:
                            landmarks = output[0, 7:].T
                            current_sequence += [landmarks[:-1]]

                        if len(current_sequence) == 10:
                            current_sequence = np.array([current_sequence])
                            payload = {'array': current_sequence.tolist() }
                            response = predict_http_request(payload)

                            current_score = response['score']

                            previous_status = current_status
                            current_status = response['status']
                            # score, status = asyncio.run(predict_request(payload))
                            # if status == 'server-error':
                            #     print('Server error or server not launched')
                            # print(score, status)
                            current_sequence = []

                        # if current_status == previous_status:
                        #     if not bad_audio_thread.is_alive() and longevity < 30:
                        #         longevity += 1
                        #     else:
                        #         longevity = 0
                        # else:
                        #     longevity = 0

                        # if longevity == 30 and current_status == "bad":
                        #     try:
                        #         if not bad_audio_thread.is_alive():
                        #             bad_audio_thread = threading.Thread(target=play_audio("bad"))
                        #             bad_audio_thread.start()
                        #     except Exception as e:
                        #         pass




                im0 = image[0].permute(1, 2, 0) * 255 # Change format [b, c, h, w] to [h, w, c] for displaying the image.
                im0 = im0.cpu().numpy().astype(np.uint8)
                
                im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR) #reshape image format to (BGR)
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                for i, pose in enumerate(output_data):  # detections per image
                    if empty: break
                
                    if len(output_data) == 0:
                        continue
                    for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose[:,:6])): #loop over poses for drawing on frame
                        c = int(cls)  # integer class
                        kpts = pose[det_index, 6:]


                        if multiple:
                            # get the centroid (cx, cy) for the current rectangle
                            rect = [tensor.cpu().numpy() for tensor in xyxy]
                            cx, cy = (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2
                            matched_object_id = None

                            # iterating through known people
                            for object_id, data in people.items():
                                distance = np.sqrt((cx - data['centroid'][0]) ** 2 + (cy - data['centroid'][1]) ** 2)
                                print(distance)
                                if distance < 300:  # Adjust the threshold as needed
                                    matched_object_id = object_id
                                    break

                            if matched_object_id is None:
                                matched_object_id = next_object_id
                                next_object_id += 1

                            if matched_object_id not in people:
                                people[matched_object_id] = {'centroid': (cx, cy), 'yoloid': det_index, 'status': 'good', 'score': 0, 'sequence' : []}
                            else:
                                people[matched_object_id]['centroid'] = (cx, cy)
                                people[matched_object_id]['yoloid'] = det_index

                            obj = people[matched_object_id]
                            label = f"ID #{obj['yoloid']} Score: {obj['score']:.2f}"
                            plot_one_box_kpt(xyxy, im0, label=label, color=colors(c, True), 
                                        line_thickness=3, kpt_label=True, kpts=kpts, steps=3, 
                                        cmap=people[matched_object_id]['status'])
                        else:
                            label = f"ID #{0} Score: {current_score:.2f}"
                            plot_one_box_kpt(xyxy, im0, label=label, color=colors(c, True), 
                                        line_thickness=3,kpt_label=True, kpts=kpts, steps=3, 
                                        cmap=current_status)


                if frame_count % separation == 0 and multiple:
                    for _, data in people.items():
                        if data['yoloid'] < output.shape[0]:
                            yoloid = data['yoloid']
                            landmarks = output[yoloid, 7:].T
                            data['sequence'] += [landmarks[:-1]]
                        
                            if len(data['sequence']) == length:
                                payload = {'array': np.array([data['sequence']]).tolist()}
                                response = predict_http_request(payload)

                                data['score'] = response['score']
                                data['status'] = response['status']
                                data['sequence'] = []

                            # print(f"{data['yoloid']} -> {data['status']}", end=' ')
                        else:
                            data['sequence'] = []

                    statuses = [(people[p]['yoloid'], people[p]['status']) for p in people]
                    # for id, status in statuses:
                    #     print(f'{id}: {status}', end='\t')
                    # print()


                frame_count += 1

                
                cv2.imshow("YOLOv7 Pose Estimation Demo", im0)
                key = cv2.waitKey(1) & 0xFF  # Wait for 1 millisecond and get the pressed key
                if key == ord('q'):
                    cv2.destroyAllWindows()  # Close the window if 'q' is pressed
                    break
            else:
                break

        cap.release()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', type=str, default='video/0', help='video/0 for webcam') #video source
    parser.add_argument('-d', '--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)')   #device arugments
    parser.add_argument('-sep', '--separation', type=str, default='1', help='Each how many frames the prediction will be executed. Defaults to 1, increase for performance') #separation arugments
    parser.add_argument('-l', '--length', type=str, default='10', help='Defines the length of the sequence. Defaults to 10, decrease for performance') #separation arugments
    parser.add_argument('-m', '--multiple', default=False, action='store_true', help='Enable multiple-person detection')  # Boolean for multiple person detection
    opt = parser.parse_args()
    return opt
    

def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    strip_optimizer(opt.device, POSEWEIGHTS)
    main(opt)
