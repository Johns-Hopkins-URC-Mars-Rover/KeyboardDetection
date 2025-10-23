import cv2
from cv2 import aruco
import argparse
import pickle
import os

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir", required=True,
	help="path to the calibration photos")
args = vars(ap.parse_args())

# ChAruco board variables
CHARUCOBOARD_ROWCOUNT = 11
CHARUCOBOARD_COLCOUNT = 8
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_4X4_250)

# Create constants to be passed into OpenCV and Aruco methods
CHARUCO_BOARD = aruco.CharucoBoard_create(
	squaresX=CHARUCOBOARD_COLCOUNT,
	squaresY=CHARUCOBOARD_ROWCOUNT,
	squareLength=0.015,
	markerLength=0.011,
	dictionary=ARUCO_DICT)

# Corners discovered in all images processed
corners_all = []

# Aruco ids corresponding to corners discovered 
ids_all = [] 

# Determined at runtime
image_size = None 

image_directory = args['dir']

for filename in os.listdir(image_directory):
	img = cv2.imread(os.path.join(image_directory, filename))

	# Grayscale the image
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Find aruco markers in the query image
	corners, ids, _ = aruco.detectMarkers(
		image=gray,
		dictionary=ARUCO_DICT)
	
	# If none found, take another capture
	if ids is None:
		continue
	
	# Outline the aruco markers found in our query image
	img = aruco.drawDetectedMarkers(
		image=img, 
		corners=corners)

	# Get charuco corners and ids from detected aruco markers
	response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
		markerCorners=corners,
		markerIds=ids,
		image=gray,
		board=CHARUCO_BOARD)

	# If a Charuco board was found, collect image/corner points
	# Requires at least 20 squares for a valid calibration image
	if response > 20:
		# Add these corners and ids to our calibration arrays
		corners_all.append(charuco_corners)
		ids_all.append(charuco_ids)
		
		# Draw the Charuco board we've detected to show our calibrator the board was properly detected
		# img = aruco.drawDetectedCornersCharuco(
		# 	image=img,
		# 	charucoCorners=charuco_corners,
		# 	charucoIds=charuco_ids)
	
		# # If our image size is unknown, set it now
		if not image_size:
			image_size = (8 * 15 + 2 * 80, 11 * 15 + 2 * 25)
		
		# # Reproportion the image, maxing width or height at 1000
		proportion = max(img.shape) / 1000.0
		img = cv2.resize(img, (int(img.shape[1]/proportion), int(img.shape[0]/proportion)))

		# # Pause to display each image, waiting for key press
		cv2.imshow('Charuco board', img)
		if cv2.waitKey(0) == ord('q'):
			break


# Destroy any open CV windows
cv2.destroyAllWindows()

# Show number of valid captures

# Make sure we were able to calibrate on at least one charucoboard
if len(corners_all) == 0:
	print("Calibration was unsuccessful. We couldn't detect charucoboards in the video.")
	print("Make sure that the calibration pattern is the same as the one we are looking for (ARUCO_DICT).")
	exit()
print("Generating calibration...")

# Now that we've seen all of our images, perform the camera calibration
calibration, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
	charucoCorners=corners_all,
	charucoIds=ids_all,
	board=CHARUCO_BOARD,
	imageSize=image_size,
	cameraMatrix=None,
	distCoeffs=None)
		
# Print matrix and distortion coefficient to the console
print("Camera intrinsic parameters matrix:\n{}".format(cameraMatrix))
print("\nCamera distortion coefficients:\n{}".format(distCoeffs))
		
# Save the calibrationq
f = open('./CameraCalibration.pckl', 'wb')
pickle.dump((cameraMatrix, distCoeffs, rvecs, tvecs), f)
f.close()
		
# Print to console our success
print('Calibration successful. Calibration file created: {}'.format('CameraCalibration.pckl'))

f = open('/home/clara/Downloads/CameraCalibration.pckl', 'rb')
(cameraMatrix, distCoeffs, _, _) = pickle.load(f, encoding='latin1')
f.close()
print("image_size:", image_size)          # should be (width, height)
print("image shape (example):", img.shape) # (h, w, c)
print("K:", cameraMatrix)
w,h = image_size
print("image center:", (w/2, h/2))

test_img = cv2.imread(os.path.join(image_directory, os.listdir(image_directory)[0]))
undistorted = cv2.undistort(test_img, cameraMatrix, distCoeffs)
proportion = max(undistorted.shape) / 1000.0
undistorted = cv2.resize(undistorted, (int(undistorted.shape[1]/proportion), int(undistorted.shape[0]/proportion)))

cv2.imshow("Undistorted", undistorted)
cv2.waitKey(0)
cv2.destroyAllWindows()
