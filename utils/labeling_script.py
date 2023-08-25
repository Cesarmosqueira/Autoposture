import cv2
import pandas as pd
import numpy as np
dataset = pd.read_csv('assets/extracted_landmarks_test.csv')

def is_float(num):
    try:
        float(num)
        return True
    except ValueError:
        return False
    
dataset['landmarks'] = dataset['landmarks'].apply(lambda arr: np.array([float(n) for n in arr.split() if is_float(n)]))

video_dir = 'assets/dataset_videos/batch_test'  # Directory where your video files are stored

dataset['Label'] = None
for index, row in dataset.iterrows():
    video = row['video']
    frame = row['frame']

    # Construct the path to the video file
    video_path = video_dir + video  # Update the file extension if necessary

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the frames per second (FPS) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"{video} at group {row['group']}. Frame {frame}")
    # Read and discard frames until the desired frame index
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)

    # Display ~10 frames
    for _ in range(10):
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Video', frame)

    # Ask for user input
    key = cv2.waitKey(0) & 0xFF
    if key == ord('g'):
        dataset.at[index, 'Label'] = 'good'  # Update 'Label' to 'good' in the corresponding row
        print("Label set to 'good'")
    elif key == ord('b'):
        dataset.at[index, 'Label'] = 'bad'  # Update 'Label' to 'bad' in the corresponding row
        print("Label set to 'bad'")

    # Release the video capture
    cap.release()

    # Close the video display window
    cv2.destroyAllWindows()

dataset.to_csv('assets/labeled_dataset_test.csv', index=False)
print(dataset.head())
