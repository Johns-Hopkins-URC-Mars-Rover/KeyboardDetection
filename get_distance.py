import cv2
import numpy as np
import pickle

MARKER_SIZE = 0.011 # in m

img = cv2.imread("/home/clara/Downloads/test-1.png")

f = open('/home/clara/Downloads/CameraCalibration.pckl', 'rb')
(cameraMatrix, distCoeffs, _, _) = pickle.load(f, encoding='latin1')
f.close()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters()

print("Camera Matrix:")
print(cameraMatrix)
print("Dist Coefficients:")
print(distCoeffs)

# Detect the markers
(corners, ids, rejected) = cv2.aruco.detectMarkers(gray, aruco_dict)
(rvec , tvec, _) = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, cameraMatrix, distCoeffs)

# Print the detected markers
print("Detected markers:")
i = 0
j = 0
for id in ids:
    print(id, end=" ")
    print(tvec[j][0])
    i += 1
    if i == 8:
        i = 0
        print()
    j += 1
print()

if ids is not None:
    cv2.aruco.drawDetectedMarkers(img, corners, ids)
    cv2.imshow('Detected Markers', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
