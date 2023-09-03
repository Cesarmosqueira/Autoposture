# Autoposture
Video-based Musculoskeletal Risk Assessment: Predictive Software for Evaluating Musculoskeletal Disorders

## Important:
- Clone this repo with --recurse-submodules
- Download the Pose Estimation pre-trained model from [here](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt)
- Download the autoposture pre-trained module latest release from [releases](https://github.com/Cesarmosqueira/Autoposture/releases)
Or copy and paste
```
git clone https://github.com/Cesarmosqueira/Autoposture --recurse-submodules
cd Autoposture
wget -P src_models/ https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt
wget -P src_models/ https://github.com/Cesarmosqueira/Autoposture/releases/download/v1.0.0/autoposture_v1.0.0.h5
```

## Run:
```
python autoposture.py --source /dev/video2 --view-img --device cuda
```
