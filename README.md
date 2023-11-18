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


## Before running the live demo
Check the [prediction system api](https://github.com/Cesarmosqueira/Autoposture/tree/master/prediction_api#readme)


## Run:
This is a default example.
```
docker pull cesarmosqueira/autoposture_api
docker-compose -f prediction_api/docker-compose.yaml up
pip install -r requirements.txt
python start.py
```
 - view-img: enables the live display

#### Remember
- To list the available webcam sources you can use `v4l2-ctl --list-devices`
- Is neccesary to run the docker image (the api)
