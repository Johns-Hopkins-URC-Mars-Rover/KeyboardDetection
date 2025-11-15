import cv2
import numpy as np
import time
from datetime import datetime
import subprocess   # for running external script

# Config
DICT = cv2.aruco.DICT_4X4_250   # match your printed tags
CAM_INDEX = 0                   # 0 is the default webcam
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
EXPECTED_IDS = {1, 2, 3, 4}
STABLE_DURATION = 10  # seconds all markers must be visible continuously

# Calibration
USE_POSE = False
K = None
dist = None
MARKER_SIZE = 2 # in cm

# --- Setup camera ---
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Check permissions/camera index.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)

# --- ArUco dictionary & detector ---
aruco_dict = cv2.aruco.getPredefinedDictionary(DICT)
use_new = hasattr(cv2.aruco, "ArucoDetector")

if use_new:
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
else:
    params = cv2.aruco.DetectorParameters_create()

# --- Main loop ---
prev = time.time()
keyboard_launched = False
all_detected_start_time = None

while True:
    ok, frame = cap.read()
    if not ok:
        print("Frame grab failed.")
        break

    # Detect markers
    if use_new:
        corners, ids, rejected = detector.detectMarkers(frame)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=params)
    
    rvec , tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, mtx, dist)


    out = frame.copy()
    detected_ids = set()

    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(out, corners, ids)
        detected_ids = set(int(i) for i in ids.flatten())


        # Overlay IDs
        for i, c in enumerate(corners):
            c = c[0].astype(int)
            cx, cy = c.mean(axis=0).astype(int)
            cv2.putText(out, f"id:{int(ids[i])}", (cx, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        # --- Check if ALL expected tags are present ---
        if EXPECTED_IDS.issubset(detected_ids):
            if all_detected_start_time is None:
                all_detected_start_time = time.time()  # start timing
            elif (time.time() - all_detected_start_time >= STABLE_DURATION) and not keyboard_launched:
                print("✅ All markers stable for 10s! Launching KeyboardDetection.py...")
                subprocess.Popen(["python3", "KeyboardDetection.py"])
                keyboard_launched = True
        else:
            # Reset timer if markers are lost
            all_detected_start_time = None
        if EXPECTED_IDS.issubset(detected_ids) and not keyboard_launched:
            print("All markers detected! Launching keyboard_detection.py...")
            subprocess.Popen(["python3", "KeyboardDetection.py"]) 
            keyboard_launched = True  # prevent relaunch
    else:
        cv2.putText(out, "No markers", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        all_detected_start_time = None

    # Display timer progress (optional)
    if all_detected_start_time is not None and not keyboard_launched:
        elapsed = time.time() - all_detected_start_time
        cv2.putText(out, f"Stable: {elapsed:.1f}s / {STABLE_DURATION}s",
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    # FPS
    now = time.time()
    fps = 1.0 / (now - prev)
    prev = now
    cv2.putText(out, f"FPS: {fps:.1f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("ArUco Live", out)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        fname = f"aruco_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(fname, out)
        print(f"Saved {fname}")

cap.release()
cv2.destroyAllWindows()
