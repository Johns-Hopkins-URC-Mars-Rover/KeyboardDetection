import cv2
import numpy as np

# pick your dictionary (common: 4x4_50, 5x5_100, 6x6_250, ARUCO_ORIGINAL)
DICT = cv2.aruco.DICT_4X4_250

# load image
img = cv2.imread("input.jpg")
if img is None:
    raise SystemExit("Couldn't read input.jpg")

# get dictionary + default detector params
aruco_dict = cv2.aruco.getPredefinedDictionary(DICT)

# OpenCV 4.7+ uses ArucoDetector; older 4.x used detectMarkers()
use_new = hasattr(cv2.aruco, "ArucoDetector")
if use_new:
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, rejected = detector.detectMarkers(img)
else:
    params = cv2.aruco.DetectorParameters_create()
    corners, ids, rejected = cv2.aruco.detectMarkers(img, aruco_dict, parameters=params)

# draw detections
out = img.copy()
if ids is not None and len(ids) > 0:
    cv2.aruco.drawDetectedMarkers(out, corners, ids)

    # OPTIONAL: pose estimation if you know camera intrinsics + marker size
    # Fill these with your calibration results:
    #   K: 3x3 camera matrix, dist: 1x5 (or 1x8) distortion coeffs
    #   marker_size_m: marker side length in meters
    # K = np.array([[fx, 0, cx],
    #               [0, fy, cy],
    #               [0,  0,  1]], dtype=np.float32)
    # dist = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
    # marker_size_m = 0.03
    # rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size_m, K, dist)
    # for rvec, tvec in zip(rvecs, tvecs):
    #     cv2.drawFrameAxes(out, K, dist, rvec, tvec, marker_size_m * 0.5)

else:
    print("No markers found.")

cv2.imwrite("output_with_markers.jpg", out)
print("Saved: output_with_markers.jpg")