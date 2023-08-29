# Autoposture
Video-based Musculoskeletal Risk Assessment: Predictive Software for Evaluating Musculoskeletal Disorders


check this out BRO https://github.com/retkowsky/Human_pose_estimation_with_YoloV7/blob/main/Human_pose_estimation_YoloV7.ipynb

## Guide:
```
cd vendor
git clone https://github.com/RizwanMunawar/yolov7-pose-estimation.git
```
## Run:
```
python3 live-testing.py --source 0 --device 0 --view-img
```

## To execute data processing

After unzipping the dataset.zip inside `dataset_videos`...

First run the [landmark extraction notebook](landmark_extraction_mechanism/our-estimation.ipynb)
that will update the `dataset.csv` in the same folder

To label the data run the [Labeling script](labeling_script.py). Press <kbd>G</kbd> or <kbd>B</kbd> to set a sequence as Good or Bad.

That will create a `labeled_dataset.csv` inside the landmark_extraction_mechanism folder
