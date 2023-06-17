#!/usr/bin/env python
# coding: utf-8

# In[6]:


import time

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import ipywidgets as widgets
from scipy.spatial.transform import Rotation

import mediapipe as mp


# In[9]:


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Choose the desired video codec (e.g., 'mp4v', 'xvid', 'MJPG')
output_file = 'testing.mp4'  # Specify the output video file name
frame_width = 640  # Specify the width of the frames in the output video
frame_height = 480  # Specify the height of the frames in the output video
frame_rate = 30.0  # Specify the frame rate of the output video

# Create a VideoWriter object
out = cv2.VideoWriter(output_file, fourcc, frame_rate, (frame_width, frame_height))


cap = cv2.VideoCapture(0)  # Change the index to use a different camera

raw_landmarks = []


# Initialize variables for FPS calculation
prev_time = time.time()
frame_count = 0
fps_array = []
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        # Perform pose estimation
        results = pose.process(frame_rgb)

        # Extract the pose landmarks
        if results.pose_landmarks:
            raw_landmarks += [results.pose_landmarks]


        # Draw the pose landmarks on the frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        
        
        # Calculate the current FPS
        current_time = time.time()
        frame_count += 1
        fps = frame_count / (current_time - prev_time)

        # Add FPS text overlay to the frame
        fps_text = f'FPS: {fps:.2f}'
        fps_array +=  [fps]
        
        
        # Calculate the position and size of the text overlay
        text_width, h = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        text_width = int(0.2 * frame_width)
        text_height = h
        text_pos = (frame_width - text_width - 10, frame_height - text_height - 10)

        # Add the FPS text to the frame
        cv2.putText(frame, fps_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        
        
        out.write(frame)
        cv2.imshow('MediaPipe Pose Estimation', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
out.release()
cv2.destroyAllWindows()
print("Window destroyed")

frame_indices = range(len(fps_array))

# Create a figure and axis object
fig, ax = plt.subplots()

# Plot the fps values
ax.plot(frame_indices, fps_array, color='blue', marker='o', linestyle='-')

# Set labels and title
ax.set_xlabel('Frame Index')
ax.set_ylabel('FPS')
ax.set_title('Frames per Second (FPS)')

# Customize grid lines
ax.grid(True, linestyle='--', alpha=0.5)

# Add a fancy background color
ax.set_facecolor('#f0f0f0')

# Add additional styling if desired
# For example, you can adjust font size, line thickness, etc.

# Show the plot
plt.show()

def parse_landmarks(landmarks):
    new_landmarks = []
    visibles = []
    for landmark in landmarks:
        new_landmarks += [[[p.x, p.y, p.z] for p in landmark.landmark]]
        visibles += [[p.visibility for p in landmark.landmark]]
    
    rotation_x = np.radians(1)  # rotation around x-axis (in radians)
    rotation_y = np.radians(90)  # rotation around y-axis (in radians)
    rotation_z = np.radians(1)  # rotation around z-axis (in radians)

    rot_matrix_x = np.array([
        [1, 0, 0],
        [0, np.cos(rotation_x), -np.sin(rotation_x)],
        [0, np.sin(rotation_x), np.cos(rotation_x)]
    ])

    rot_matrix_y = np.array([
        [np.cos(rotation_y), 0, np.sin(rotation_y)],
        [0, 1, 0],
        [-np.sin(rotation_y), 0, np.cos(rotation_y)]
    ])

    rot_matrix_z = np.array([
        [np.cos(rotation_z), -np.sin(rotation_z), 0],
        [np.sin(rotation_z), np.cos(rotation_z), 0],
        [0, 0, 1]
    ])

    # Apply rotation to each frame of landmarks
    rotated_landmarks_list = []
    for frame_landmarks in new_landmarks:
        frame_landmarks = np.array(frame_landmarks)
        rotated_landmarks = np.dot(rot_matrix_z, np.dot(rot_matrix_y, np.dot(rot_matrix_x, frame_landmarks.T))).T
        rotated_landmarks_list.append(rotated_landmarks.tolist())
    return np.array(new_landmarks), np.array(visibles)


# In[80]:


landmarks, visibles = parse_landmarks(raw_landmarks)


# In[81]:


axis = np.array([0, 1, 0]) 
angular_speed = 3
rotation_angle = np.deg2rad(angular_speed)

rotation = Rotation.from_rotvec(rotation_angle * axis)


# In[82]:


def plot_person(index):
    global rotation_angle
    ax.clear()
    
    frame_final = landmarks[index]

    visibility = visibles[index]
    minimum_visibility = 0.5
    
    # Plot the 3D body lines
    for connection in mp_pose.POSE_CONNECTIONS:
        joint1 = connection[0]
        joint2 = connection[1]
        if visibility[joint1] > minimum_visibility and visibility[joint2] > minimum_visibility:
            x = [frame_final[joint1][0], frame_final[joint2][0]]
            y = [frame_final[joint1][1], frame_final[joint2][1]]
            z = [frame_final[joint1][2], frame_final[joint2][2]]

            ax.plot(x, y, z, c='r')
        
    # Plot the 3D body landmarks as dots
    x = [landmark[0] for i, landmark in enumerate(frame_final) if visibility[i] > minimum_visibility]
    y = [landmark[1] for i, landmark in enumerate(frame_final) if visibility[i] > minimum_visibility]
    z = [landmark[2] for i, landmark in enumerate(frame_final) if visibility[i] > minimum_visibility]
    ax.scatter(x, y, z, c='b')
    
    # Set the plot limits
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    
    return ax


# In[83]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plot_person(80)


# In[84]:


landmarks.shape


# In[85]:


plt.rcParams['animation.embed_limit'] = 50.0  # Increase the embedding limit to 50 MB

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
def update(index):
    ax.cla()
    plot_person(index)

# Create the animation
animation = FuncAnimation(fig, update, frames=len(landmarks), interval=15)

# Display the animation in Jupyter Notebook


# In[60]:


# Set up the video writer
writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)

# Save the animation as a video
animation.save('animation.mp4', writer=writer)
