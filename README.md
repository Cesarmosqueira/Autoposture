# Autoposture
Video-based Musculoskeletal Risk Assessment: Predictive Software for Evaluating Musculoskeletal Disorders

## Important:
- Clone this repo with --recurse-submodules
- Download the Pose Estimation pre-trained model from [here](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt) or copy and paste:
```
wget -P src_models/ https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt
```

## Run:
```
python autoposture.py --source /dev/video2 --view-img --device cuda
```
