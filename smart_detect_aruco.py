import cv2
import numpy as np
import time
from datetime import datetime
import subprocess, sys, os

# --- Config (relaxed + stable) ---
DICT = cv2.aruco.DICT_4X4_250
CAM_INDEX = 0
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

RELAX_MODE = True  # ID-agnostic; accepts any 3+ tags
EXPECTED_IDS = {1, 2, 3, 4}
TAG_TO_CORNER = {1: "TL", 2: "TR", 3: "BR", 4: "BL"}

MIN_AREA_PER_TAG = 400
MIN_BOARD_AREA_PCT = 0.015
STABLE_FRAMES_N = 5
MAX_WAIT_FRAMES = 1800

PREVIEW_W, PREVIEW_H = 900, 300

# ---- Path to your YOLO script (absolute path recommended) ----
KEYBOARD_DET_PATH = os.path.abspath("KeyboardDetection.py")  # adjust if needed

# --- Camera ---
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Check permissions/camera index.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)

# --- ArUco detector (tolerant) ---
aruco_dict = cv2.aruco.getPredefinedDictionary(DICT)
use_new = hasattr(cv2.aruco, "ArucoDetector")

if use_new:
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 4
    params.adaptiveThreshConstant = 7
    params.minMarkerPerimeterRate = 0.015
    params.maxMarkerPerimeterRate = 5.0
    params.polygonalApproxAccuracyRate = 0.04
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 7
    params.cornerRefinementMaxIterations = 50
    params.cornerRefinementMinAccuracy = 0.01
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
else:
    params = cv2.aruco.DetectorParameters_create()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 4
    params.adaptiveThreshConstant = 7
    params.minMarkerPerimeterRate = 0.015
    params.maxMarkerPerimeterRate = 5.0
    params.polygonalApproxAccuracyRate = 0.04
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 7
    params.cornerRefinementMaxIterations = 50
    params.cornerRefinementMinAccuracy = 0.01

# --- Helpers ---
def tag_centers(corners, ids):
    centers, areas = {}, {}
    if ids is None:
        return centers, areas
    for c, i in zip(corners, ids.flatten()):
        pts = c[0].astype(np.float32)
        cx, cy = pts.mean(axis=0)
        centers[int(i)] = (float(cx), float(cy))
        areas[int(i)] = float(cv2.contourArea(pts))
    return centers, areas

def ordered_corners_by_ids(centers):
    need = ["TL", "TR", "BR", "BL"]
    lut = {}
    for tid, name in TAG_TO_CORNER.items():
        if tid in centers:
            lut[name] = centers[tid]
    if not all(n in lut for n in need):
        return None
    return np.array([lut[n] for n in need], dtype=np.float32)

def ordered_corners_geo(centers):
    pts = np.array(list(centers.values()), dtype=np.float32)
    if pts.shape[0] < 3:
        return None
    hull = cv2.convexHull(pts)
    if len(hull) < 3:
        return None
    hull = hull.reshape(-1, 2)
    if hull.shape[0] == 3:
        dists = [np.linalg.norm(hull[(i+1)%3] - hull[i]) for i in range(3)]
        i_long = int(np.argmax(dists))
        p_mid = (hull[i_long] + hull[(i_long+1)%3]) / 2.0
        hull = np.vstack([hull, p_mid])
    s = hull.sum(axis=1)
    d = (hull[:,0] - hull[:,1])
    tl = hull[np.argmin(s)]
    br = hull[np.argmax(s)]
    tr = hull[np.argmin(d)]
    bl = hull[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def homography_ok(pts_img, w=PREVIEW_W, h=PREVIEW_H):
    src = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32)
    H, _ = cv2.findHomography(pts_img, src, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    if H is None:
        return False, None
    return True, H

def view_quality(frame, centers, areas):
    Hf, Wf = frame.shape[:2]
    frame_area = float(Hf * Wf)
    detected_ids = set(centers.keys())

    if RELAX_MODE:
        if len(detected_ids) < 3:
            return False, "need >=3 tags"
        ids_sorted = sorted(detected_ids, key=lambda i: areas.get(i, 0.0), reverse=True)
        top_ids = ids_sorted[:4]
        if any(areas.get(i, 0.0) < MIN_AREA_PER_TAG for i in top_ids):
            return False, "tags too small"
        pts = ordered_corners_geo({i: centers[i] for i in top_ids})
        if pts is None:
            return False, "ordering fail"
    else:
        if not EXPECTED_IDS.issubset(detected_ids):
            return False, "missing expected ids"
        if any(areas.get(i, 0.0) < MIN_AREA_PER_TAG for i in EXPECTED_IDS):
            return False, "tags too small"
        pts = ordered_corners_by_ids(centers)
        if pts is None:
            return False, "corner map incomplete"

    board_area = float(cv2.contourArea(pts))
    if board_area / frame_area < MIN_BOARD_AREA_PCT:
        return False, "board too small"

    okH, _ = homography_ok(pts)
    if not okH:
        return False, "homography bad"

    return True, "ok"

# --- Main loop ---
prev = time.time()
stable_frames = 0
wait_frames = 0

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
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(out, corners, ids)
        centers, areas = tag_centers(corners, ids)

        # Debug overlays
        detected_ids_sorted = sorted(list(centers.keys()))
        y0 = 160
        cv2.putText(out, f"Detected IDs: {detected_ids_sorted}", (20, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        y = y0 + 24
        for tid in detected_ids_sorted[:8]:
            cv2.putText(out, f"id {tid} area: {int(areas.get(tid,0))} px", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,255,180), 1, cv2.LINE_AA)
            y += 18

        good, reason = view_quality(frame, centers, areas)
        color = (0, 200, 0) if good else (0, 0, 255)
        cv2.putText(out, f"VIEW: {'OK' if good else 'WAIT'} ({reason})",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        if good:
            stable_frames += 1
        else:
            stable_frames = 0
        wait_frames += 1

        # progress bar
        bar_w = int(200 * min(1.0, stable_frames / float(STABLE_FRAMES_N)))
        cv2.rectangle(out, (20, 60), (220, 80), (60,60,60), -1)
        cv2.rectangle(out, (20, 60), (20 + bar_w, 80), (0, 200, 0), -1)
        cv2.putText(out, f"Stable {stable_frames}/{STABLE_FRAMES_N}",
                    (230, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1, cv2.LINE_AA)

        # SMART TRIGGER — use same interpreter, release camera, close windows, then exit.
        if stable_frames >= STABLE_FRAMES_N:
            print("✅ Stable view achieved. Launching KeyboardDetection.py ...")

            # Release all resources BEFORE spawning the next process
            cap.release()
            cv2.destroyAllWindows()

            # Use the exact same Python interpreter that has ultralytics installed
            python_exec = sys.executable  # e.g., /opt/anaconda3/bin/python
            # Optional: pass env so PATH/venv are identical
            env = os.environ.copy()

            # Absolute path launch for safety
            subprocess.Popen([python_exec, KEYBOARD_DET_PATH], env=env)

            # Quit this script so only YOLO owns the camera
            sys.exit(0)

        if wait_frames > MAX_WAIT_FRAMES:
            cv2.putText(out, "Timeout: move closer / reduce glare",
                        (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(out, "No markers", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        stable_frames = 0
        wait_frames += 1

    # FPS
    now = time.time()
    fps = 1.0 / max(1e-6, (now - prev))
    prev = now
    cv2.putText(out, f"FPS: {fps:.1f}", (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("ArUco Live (Relaxed Pose-Gate)", out)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        fname = f"aruco_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(fname, out)
        print(f"Saved {fname}")

cap.release()
cv2.destroyAllWindows()