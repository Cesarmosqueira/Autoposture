import sys
import os
from torchvision import transforms
import torch
import cv2
import numpy as np
import requests

yolov7_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../vendor/yolov7')
sys.path.append(yolov7_path)

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt, strip_optimizer, xyxy2xywh
from utils.plots import colors, output_to_keypoint, plot_one_box_kpt, plot_skeleton_kpts
from utils.torch_utils import select_device




HOST = 'localhost'
PORT = '8000'
def predict_http_request(payload):
    """
    Args:
        - payload: {'array': (1, 10, 50) shape (10 frames)}
    Returns:
        - score: Value between 0 and 1
        - status: Good or bad posture (depending on threshold:0.7)
    """
    response = requests.post(f"http://{HOST}:{PORT}/predict", json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        print("Error:", response.status_code)
        print(response.text)


model = None
device = None
def model_initialization(device_ref, w):
    global model, device
    device = select_device(device_ref)
    model = attempt_load(w, map_location=device)
    return model


separation =0
length = 10
frame_count = 0
# logic for multiple persons
people = {}
next_object_id = 0
# logic for single persons
current_sequence = []
current_score = 0
current_status = 'good'
previous_status = "None"
longevity = 0 # frames spent in the current status
separation = 1
multiple = False
frame_count = 0
empty = False


# for audio playing
iterations_in_bad_posture = 0 
max_iterations_in_bad_posture = 5
should_alert = False


@torch.no_grad()
def on_update(frame, recently_alerted, threshold = 0.7):
    global separation, frame_count, current_sequence, empty, current_score, current_status, should_alert,\
            iterations_in_bad_posture, max_iterations_in_bad_posture
    if recently_alerted:
        should_alert = False

    orig_image = frame 
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB) 
    image = letterbox(image, (frame.shape[1]), stride=64, auto=True)[0]
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
                response['status'] = 'good' if current_score > threshold else 'bad'
                if response['status'] == 'bad':
                    iterations_in_bad_posture += 1
                current_status = response['status']
                current_sequence = []

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
                    data['status'] = 'good' if score > THRESHOLD else 'bad'
                    data['sequence'] = []

                # print(f"{data['yoloid']} -> {data['status']}", end=' ')
            else:
                data['sequence'] = []

        
        statuses = [(people[p]['yoloid'], people[p]['status']) for p in people]
        # for id, status in statuses:
        #     print(f'{id}: {status}', end='\t')
        # print()
    if iterations_in_bad_posture >= max_iterations_in_bad_posture:
        iterations_in_bad_posture = 0
        should_alert = True

    frame_count += 1
    return im0, current_status, current_score, should_alert
