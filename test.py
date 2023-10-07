import cv2
import numpy as np

# Initialize a dictionary to store the centroids of tracked objects
centroids = {}
next_object_id = 0

# Create a VideoCapture object (0 for webcam, or provide a video file path)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection (you can replace this with your YOLOv7 pose estimation)
    # In this example, we'll use a simple method to simulate object detection
    # Replace this part with your actual pose estimation code
    # We'll use rectangles to represent people
    detections = [(100, 100, 50, 50), (200, 200, 40, 40)]  # Example detections

    # Update the centroids dictionary for tracking
    new_centroids = {}

    for rect in detections:
        x, y, w, h = rect
        cx, cy = x + w // 2, y + h // 2

        # Check if we have an existing object near this centroid
        matched_object_id = None
        for object_id, centroid in centroids.items():
            distance = np.sqrt((cx - centroid[0]) ** 2 + (cy - centroid[1]) ** 2)
            if distance < 50:  # Adjust the threshold as needed
                matched_object_id = object_id
                break

        if matched_object_id is None:
            matched_object_id = next_object_id
            next_object_id += 1

        new_centroids[matched_object_id] = (cx, cy)

        # Draw bounding box and object ID
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {matched_object_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Update the centroids dictionary
    centroids = new_centroids

    # Display the frame with tracking info
    cv2.imshow("Person Tracking", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video source and close all windows
cap.release()
cv2.destroyAllWindows()

