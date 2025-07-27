import numpy as np
import cv2 as cv
import glob
import os
import time

def dnsz(image):
    return cv.resize(image, None, fx=0.5, fy=0.5)



def calc_calib(path):
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    CW, CH = 9, 6
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((CW*CH,3), np.float32)
    objp[:,:2] = np.mgrid[0:CW,0:CH].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob(os.path.join(path,'*.jpg'))

    t0 = time.time()

    for fname in images:
        print(fname)
        img = cv.imread(fname)
        img = dnsz(img)

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        h, w = gray.shape[:2]
        if h > w:
            gray = cv.rotate(gray, cv.ROTATE_90_COUNTERCLOCKWISE)
            print('rotate')
        
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (CW,CH), cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_FAST_CHECK | cv.CALIB_CB_NORMALIZE_IMAGE)
        print(ret)
        # If found, add object points, image points (after refining them)
        if ret == True:        
            objpoints.append(objp)

            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv.drawChessboardCorners(img, (CW,CH), corners2, ret)
            cv.imshow('img', dnsz(img))
            cv.waitKey(100)
            
    t1 = time.time()

    print('corner findings', t1-t0)

    cv.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    img = cv.imread(images[0])
    img = dnsz(img)
    h,  w = img.shape[:2]
    #newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))

    t2 = time.time()

    print('calibration', t2-t1)

    # undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imwrite('calibresult.jpg', dst)

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error

    print( "total error: {}".format(mean_error/len(objpoints)) )
    
    return mtx, dist, rvecs, tvecs, newcameramtx, roi

if __name__ == "__main__":
    #images_path = "./data/chess"
    images_path = "./data/chess2"
    mtx, dist, rvecs, tvecs, newcameramtx, roi = calc_calib(images_path)
    print('camera matrix', mtx)
    print('distortions', dist)
    print('newcameramtx', newcameramtx)
    print('rvecs', rvecs)
    print('tvecs', tvecs)
    print('roi', roi)
    
    '''with open("calib.txt", "w") as file:
        file.write("\ncamera matrix\n")        
        file.write(mtx)
        file.write("\ndistortions\n")        
        file.write(dist)
        file.write("\nnewcameramtx\n")        
        file.write(newcameramtx)
        file.write("\nrvecs\n")        
        file.write(rvecs)
        file.write("\ntvecs\n")        
        file.write(tvecs)
        file.write("\nroi\n")        
        file.write(roi)
        '''
