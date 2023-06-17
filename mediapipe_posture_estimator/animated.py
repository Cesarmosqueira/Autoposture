import cv2
import mediapipe as mp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Initialize MediaPipe solutions for human pose estimation
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Set up the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create an empty list to store the coordinates of the body lines
body_lines = []

# Initialize the video capture using OpenCV
cap = cv2.VideoCapture(0)  # Change the index to use a different camera

def update(frame):
    ret, frame = cap.read()
    if not ret:
        return

    # Convert the BGR frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False

    # Perform pose estimation
    results = pose.process(frame_rgb)

    # Extract the pose landmarks
    if results.pose_landmarks:
        # Clear the previous plot
        ax.cla()

        # Plot the 3D body lines
        for connection in mp_pose.POSE_CONNECTIONS:
            joint1 = connection[0]
            joint2 = connection[1]
            x = [results.pose_landmarks.landmark[joint1].x, results.pose_landmarks.landmark[joint2].x]
            y = [results.pose_landmarks.landmark[joint1].y, results.pose_landmarks.landmark[joint2].y]
            z = [results.pose_landmarks.landmark[joint1].z, results.pose_landmarks.landmark[joint2].z]
            body_lines.append((x, y, z))
            ax.plot(x, y, z, c='r')

        # Set the plot limits
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

    # Draw the pose landmarks on the frame
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow('MediaPipe Pose Estimation', frame)

# Create the animation
animation = FuncAnimation(fig, update, interval=10, save_count=100)  # Adjust save_count as needed

# Initialize MediaPipe Pose
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    plt.show()

cap.release()
cv2.destroyAllWindows()
