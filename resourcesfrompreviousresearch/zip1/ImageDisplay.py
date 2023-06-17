import cv2
import DetectPoseFunction as dp

image = cv2.imread('media/sample2.jpg')
dp.detectPose(image, dp.pose, display = True, verbose = False)