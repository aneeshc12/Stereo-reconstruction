import cv2
import numpy as np
import os
import glob
import PIL.Image as Image
import PIL.ExifTags as ExifTags
 
# Defining the dimensions of checkerboard
CHECKERBOARD = (10,7)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = [] 
 
 
# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None
 
# Extracting path of individual image stored in a given directory
images = glob.glob('./checkerboard_images_Anis_iphone/*.jpeg')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
     
    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display 
    them on the images of checker board
    """
    if ret == True:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
         
        imgpoints.append(corners2)
 
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
     
    cv2.imshow('img',img)
    cv2.waitKey(100)
 
cv2.destroyAllWindows()
 
h,w = img.shape[:2]
 
"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

#Save parameters into numpy file
np.save("./camera_params/ret", ret)
np.save("./camera_params/K", K)
np.save("./camera_params/dist", dist)
np.save("./camera_params/rvecs", rvecs)
np.save("./camera_params/tvecs", tvecs)
#Get focal length in decimal form
np.save("./camera_params/FocalLength", K[0][0])

# Calculate the pixel distance between principal points of the camera pair
pixel_distance = np.sqrt((K[0][2] - K[1][2])**2 + (K[1][2] - K[1][2])**2)

print(pixel_distance)

# Calculate the baseline distance between the cameras
baseline_distance = 0.5 * 0.0254 * 0.5 * 0.0254 * pixel_distance / (K[0][0] + K[1][0])

print(baseline_distance)

print("Camera matrix : ", K)
print("dist : ", dist)
print("rvecs : ", rvecs)
print("tvecs : ", tvecs)
print("Focal length : ", K[0][0])
print(ret)

# Load new images
imgL = cv2.imread('Image_pairs/tableL.png',0)

# Get the image size
h, w = imgL.shape[:2]
print(h, w)