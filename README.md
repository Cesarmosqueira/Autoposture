# Autoposture
Video-based Musculoskeletal Risk Assessment: Predictive Software for Evaluating Musculoskeletal Disorders

## Important:
- Clone this repo with --recurse-submodules
- Download the Pose Estimation pre-trained model from [here](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt)
- Download the autoposture pre-trained module latest release from [releases](https://github.com/Cesarmosqueira/Autoposture/releases)

Or just run:
```
git clone https://github.com/Cesarmosqueira/Autoposture --recurse-submodules
cd Autoposture
wget -P src_models/ https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt
wget -P src_models/ https://github.com/Cesarmosqueira/Autoposture/releases/download/v1.0.0/autoposture-model.h5
```


## Pre-Execution Checklist:
- For Windows users: Ensure you run the setup and execution commands as an administrator.
- Navigate to the cloned Autoposture repository directory before proceeding.
- Have two terminal windows open, both in the repo. One for the client and one for the service


## Run the api in one terminal:
```
docker pull cesarmosqueira/autoposture_api
docker-compose -f prediction_api/docker-compose.yaml up
```
For linux users: If you can't execute the second command and you get something with:  `Permission Denied`, run it as administrator

## Run client in another terminal:
```
pip install -r requirements.txt
python start.py
```
 - view-img: enables the live display

#### Remember
- Windows Users: Run commands as administrator.
- Webcam Devices: List available devices with v4l2-ctl --list-devices.
- Ensure Docker API service is active before starting the client.
