import cv2
import time
import sys
import subprocess
from datetime import datetime
from pathlib import Path

# ---------------- CONFIG ----------------
DICT = cv2.aruco.DICT_4X4_250
CAM_INDEX = 0
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

EXPECTED_IDS = {1, 2, 3, 4}
STABLE_DURATION = 10          # seconds
ASSUMED_FPS = 30              # used for stability counter target
STABLE_FRAMES_TARGET = STABLE_DURATION * ASSUMED_FPS

# Tolerance: if markers flicker for a moment, don't reset to 0
DECAY_PER_BAD_FRAME = 3       # higher = stricter; lower = more forgiving

KEYBOARD_SCRIPT = "KeyboardDetection.py"

# Log file so you can SEE what KeyboardDetection did
LOG_PATH = Path(__file__).with_name("keyboard_launch.log")

# ---------------- CAMERA SETUP ----------------
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Check permissions/camera index.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, ASSUMED_FPS)

# ---------------- ARUCO SETUP ----------------
aruco_dict = cv2.aruco.getPredefinedDictionary(DICT)
use_new = hasattr(cv2.aruco, "ArucoDetector")

if use_new:
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
else:
    params = cv2.aruco.DetectorParameters_create()

# ---------------- MAIN LOOP ----------------
prev = time.time()
keyboard_launched = False
stable_counter = 0

print("Running with Python:", sys.executable)
print("Expected IDs:", EXPECTED_IDS)
print("Stable requirement:", f"{STABLE_DURATION}s ~ {STABLE_FRAMES_TARGET} frames")
print("Log file:", str(LOG_PATH))

while True:
    ok, frame = cap.read()
    if not ok:
        print("Frame grab failed.")
        break

    if use_new:
        corners, ids, rejected = detector.detectMarkers(frame)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=params)

    out = frame.copy()
    detected_ids = set()

    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(out, corners, ids)

        # ids shape is (N,1). Convert cleanly to python ints.
        detected_ids = set(int(x) for x in ids.flatten())

        # Overlay IDs (fixes your numpy deprecation warning)
        for i, c in enumerate(corners):
            pts = c[0].astype(int)
            cx, cy = pts.mean(axis=0).astype(int)
            marker_id = int(ids[i][0])  # <- IMPORTANT FIX
            cv2.putText(out, f"id:{marker_id}", (cx, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        all_present = EXPECTED_IDS.issubset(detected_ids)

        # Stability counter logic
        if all_present:
            stable_counter = min(STABLE_FRAMES_TARGET, stable_counter + 1)
        else:
            stable_counter = max(0, stable_counter - DECAY_PER_BAD_FRAME)

        # On-screen status
        missing = EXPECTED_IDS - detected_ids
        if all_present:
            cv2.putText(out, "All markers detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(out, f"Missing: {sorted(list(missing))}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

        # Progress bar text
        progress_sec = stable_counter / ASSUMED_FPS
        cv2.putText(out, f"Stable: {progress_sec:.1f}s / {STABLE_DURATION}s",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        # Launch KeyboardDetection once, when stable enough
        if (stable_counter >= STABLE_FRAMES_TARGET) and not keyboard_launched:
            print(f"Stability reached. Launching {KEYBOARD_SCRIPT} with {sys.executable}")
            with open(LOG_PATH, "w") as f:
                f.write(f"Launched at {datetime.now().isoformat()}\n")
                f.write(f"Launcher python: {sys.executable}\n")
                f.write(f"Working dir: {Path.cwd()}\n\n")

            # Redirect stdout/stderr to log so you can debug instantly
            log_f = open(LOG_PATH, "a")
            subprocess.Popen([sys.executable, KEYBOARD_SCRIPT],
                             stdout=log_f, stderr=log_f, cwd=str(Path(__file__).parent))
            keyboard_launched = True

    else:
        stable_counter = max(0, stable_counter - DECAY_PER_BAD_FRAME)
        cv2.putText(out, "No markers", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

    # FPS overlay
    now = time.time()
    fps = 1.0 / (now - prev) if (now - prev) > 0 else 0
    prev = now
    cv2.putText(out, f"FPS: {fps:.1f}", (20, 120),
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