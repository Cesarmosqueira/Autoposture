# Autoposture
Video-based Musculoskeletal Risk Assessment: Predictive Software for Evaluating Musculoskeletal Disorders


| Media Pipe estimation | 3D Plot animated |
|---------|---------|
| ![mptest](media/testing.gif) | ![3dplot](media/animation.gif) |


## To execute data processing

After unzipping the dataset.zip inside `dataset_videos`...

First run the [landmark extraction notebook](landmark_extraction_mechanism/our-estimation.ipynb)
that will update the `dataset.csv` in the same folder

To label the data run the [Labeling script](labeling_script.py). Press <kbd>G</kbd> or <kbd>B</kbd> to set a sequence as Good or Bad.

That will create a `labeled_dataset.csv` inside the landmark_extraction_mechanism folder
