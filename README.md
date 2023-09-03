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
python autoposture.py --source 0 --view-img --device 0
```
 - view-img: enables the live display
 - source: defines the source weather be a video or a webcam  (videos are not supported)
 - device: 0 for gpu. Also can be 'cuda' 'gpu' and all the other tensorflow supported devices

#### tip
- To list the available webcam sources you can use `v4l2-ctl --list-devices`
