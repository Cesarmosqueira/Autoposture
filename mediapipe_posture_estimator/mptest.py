from re import A
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import json

# Initialize Mediapipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize Mediapipe Drawing utilities
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

def image_preprocessing(img):
    # Convert the BGR image to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    width = int(img.shape[1] * 0.65)
    height = int(img.shape[0] * 0.65)
    img = cv2.resize(img, (width, height))

    # Apply bilateral filtering
    d = 9  # Diameter of each pixel neighborhood
    sigma_color = 75  # Standard deviation of color
    sigma_space = 75  # Standard deviation of space
    filtered_image = cv2.bilateralFilter(img, d, sigma_color, sigma_space)

    # Apply CLAHE
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # lab_image = cv2.cvtColor(filtered_image, cv2.COLOR_RGB2LAB)
    # lab_image_planes = cv2.split(lab_image)
    # lab_image_planes[0] = clahe.apply(lab_image_planes[0])
    # enhanced_lab_image = cv2.merge(lab_image_planes)
    # enhanced_image = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2RGB)

    return filtered_image



landmarks3d = []
while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        continue

    image = image_preprocessing(frame)

    # Process the image and get pose landmarks
    results = pose.process(image)

    # Draw 3D pose landmarks on the frame
    if results.pose_landmarks:

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2))

        # Get 3D landmark coordinates
        parsed_landmarks = []
        for landmark in results.pose_landmarks.landmark:
            parsed_landmarks.append((landmark.x, landmark.y, landmark.z))
        landmarks3d += [parsed_landmarks]


    # Display the resulting frame
    cv2.imshow('3D Pose Estimation', frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()


with open("mediapipe_landmarks.json", "w") as f:
    to_write = {
            "3d": landmarks3d,
    }
    
    f.write(json.dumps(to_write, indent=1))
    f.close()
    
