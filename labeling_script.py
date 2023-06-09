import cv2
import pandas as pd

dataset = pd.read_csv('landmark_extraction_mechanism/dataset.csv')

# Assuming your DataFrame is called 'df'
video_dir = 'dataset_videos/'  # Directory where your video files are stored
frame_duration = 1  # Duration of each frame in seconds

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

dataset.to_csv('landmark_extraction_mechanism/labeled_dataset.csv', index=False)
print(dataset.head())
