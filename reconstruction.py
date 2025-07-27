import numpy as np
#import cv2 as cv
import cv2
import glob
import os
import time
import matplotlib.pyplot as plt
import math

def dnsz(image):
    return cv2.resize(image, None, fx=0.5, fy=0.5)
    
    
def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel().astype("int32"))
    imgpts = imgpts.astype("int32")
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def get_and_process_img(fname, mtx, dist, newcameramtx):
    img = cv2.imread(fname)
    img = dnsz(img)    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    h,s,v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    v = clahe.apply(v)
    hsv = cv2.merge([h,s,v])
    #img[((v>170) & (s < 25)) | ((v<120) & (s < 45))] = 0    
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    gray = img
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    gray = cv2.undistort(gray, mtx, dist, None, newcameramtx)    
    return gray

def show_points3D(points, pos0=None, pos1=None):
    coords = points.transpose()
        
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(coords[0], coords[1], coords[2], c='blue', marker='o', s=5)
    
    if pos0 is not None:
        ax.scatter(*pos0, c='red', marker='x', s=10)
    if pos1 is not None:
        ax.scatter(*pos1, c='red', marker='*', s=10)
       
    ax.set_aspect('equal')
    
    plt.show()
    
def show_points3D_color(points, colors):    
    coords = points.transpose()
    colors = np.array(colors)/255.0
    colors = colors[:,::-1]
        
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(coords[0], coords[1], coords[2], c=colors, marker='o', s=5)
    
    plt.show()

def show_points3D_2(points1, points2, colors1=None, colors2=None):
    coords = points1.transpose()
    coords2 = points2.transpose()
        
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    colors1 = (np.array(colors1)/255.0)[:,::-1] if colors1 is not None else 'blue'
    colors2 = (np.array(colors2)/255.0)[:,::-1] if colors2 is not None else 'red'
    
    ax.scatter(coords[0], coords[1], coords[2], c=colors1, marker='o', s=5)
    ax.scatter(coords2[0], coords2[1], coords2[2], c=colors2, marker='o', s=5)
    
    ax.set_aspect('equal')
    
    plt.show()
    
# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
 
    assert(isRotationMatrix(R))
 
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
 
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])

def main():
    images_path = "./data/chess2"
    
    mtx = np.array([[1.61976266e+03, 0.00000000e+00, 1.03544002e+03],
            [0.00000000e+00, 1.62089834e+03, 7.72818235e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    dist = np.array([[ 0.14538323, -0.31737218, -0.00070913, -0.00119763, 0.12225343]])
    rvecs, tvecs = None, None
    newcameramtx = np.array([[1.64350977e+03, 0.00000000e+00, 1.03233371e+03],
                                [0.00000000e+00, 1.65042053e+03, 7.71463391e+02],
                                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    #roi = (22, 20, 2032, 1517)
    roi = (0, 0, 2080, 1560)
    
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    CW, CH = 9, 6
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((CW*CH,3), np.float32)
    objp[:,:2] = np.mgrid[0:CW,0:CH].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob(os.path.join(images_path,'*.jpg'))

    axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

    translations = []
    origins = []

    for i in range(len(images)-1):
    #for i in range(3):
        print(images[i])
        img1 = get_and_process_img(images[i], mtx, dist, newcameramtx)  
        
        #img2 = get_and_process_img(imgaes[i+1])
        
        # Find the chess board corners
        ret, corners1 = cv2.findChessboardCorners(img1, (CW,CH), cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE)
        #ret, corners2 = cv2.findChessboardCorners(img2, (CW,CH), cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        print('OBJP', objp.shape, objp)
        print('CORNERS', corners1.shape, corners1)
        ret,rvecs, tvecs = cv2.solvePnP(objp, corners1, newcameramtx, dist)
 
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, newcameramtx, dist)
        
        #origin, jac = cv2.projectPoints(np.float32([[0,0,0]]).reshape(-1,3), rvecs, tvecs, newcameramtx, dist)
        
        translations.append(tvecs.T[0])
        R, _ = cv2.Rodrigues(rvecs)
        euler = rotationMatrixToEulerAngles(R)
        print(R)
        print(euler)
        #forward = R[:,2]
        #forward = R[2]
        forward = R.T @ np.float32([0,0,1])
        print(forward)
        origin = (-R.T @ tvecs).T[0]
        print('origin', origin)
        origins.append(origin)
        frw = (origin + forward*0.2)
        print('frw', frw)
        origins.append(frw)
        
        
        img = draw(img1,corners1,imgpts)
        #cv.imshow('img',img)
        #cv.waitKey()
        exit()
    
    print(origins)
    #show_points3D(np.float32(translations), np.zeros((3,1)))
    #show_points3D(np.float32(origins), np.zeros((3,1)))
    show_points3D_2(np.float32(origins), objp)
    
def get_cam_pos_and_dir(R, t):
    # R: rotation matrix 3x3
    # t: translation vector
    forward = R.T @ np.float32([0,0,1])
    origin = (-R.T @ t).T[0]        
    return origin, forward
    
def main2():
    images_path = "./data/chess2"
    
    mtx = np.array([[1.61976266e+03, 0.00000000e+00, 1.03544002e+03],
            [0.00000000e+00, 1.62089834e+03, 7.72818235e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    dist = np.array([[ 0.14538323, -0.31737218, -0.00070913, -0.00119763, 0.12225343]])
    rvecs, tvecs = None, None
    newcameramtx = np.array([[1.64350977e+03, 0.00000000e+00, 1.03233371e+03],
                                [0.00000000e+00, 1.65042053e+03, 7.71463391e+02],
                                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    #roi = (22, 20, 2032, 1517)
    roi = (0, 0, 2080, 1560)
    
    CW, CH = 9, 6
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((CW*CH,3), np.float32)
    objp[:,:2] = np.mgrid[0:CW,0:CH].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob(os.path.join(images_path,'*.jpg'))

    axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

    translations = []
    #origins = []
    rec_origins = []
    points3D = None

    for i in range(len(images)-1):
    #for i in range(3):
        origins = []
        print(images[i], images[i+1])
        img1 = get_and_process_img(images[i], mtx, dist, newcameramtx)
        img2 = get_and_process_img(images[i+1], mtx, dist, newcameramtx)
        #cv2.imshow('im1', img1)
        #cv2.imshow('im2', img2)
        
        # Find the chess board corners
        ret, corners1 = cv2.findChessboardCorners(img1, (CW,CH), cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE)
        ret, corners2 = cv2.findChessboardCorners(img2, (CW,CH), cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        if points3D is None:
            #print(corners1)
            corners1 = corners1.squeeze()
            corners2 = corners2.squeeze()
            assert len(corners1.shape) == 2 and corners1.shape[1] == 2, "wrong input coordinates for pose recovery"
            E, mask = cv2.findEssentialMat(corners1, corners2, newcameramtx, cv2.RANSAC, 0.999, 1.0)
            res, R, t, mask_pose = cv2.recoverPose(E, corners1, corners2, newcameramtx)    
            #print(corners1)
            print('all pts', len(corners1))
            print('recover', res)
            #print('mask', mask)
            
            R0 = np.eye(3)
            t0 = np.zeros((3,1))
            P1 = np.hstack((R0, t0))
            P2 = np.hstack((R, t0+t))        

            P1 = newcameramtx @ P1
            P2 = newcameramtx @ P2

            inliers = mask_pose.ravel().astype(bool)
            
            pts1_in = corners1[inliers].T
            pts2_in = corners2[inliers].T
            print(pts1_in.shape)

            homog_pts4d = cv2.triangulatePoints(P1, P2, pts1_in, pts2_in)        
            points3D = cv2.convertPointsFromHomogeneous(homog_pts4d.T)
            print('homog pts', homog_pts4d.shape)
            #points3D = (homog_pts4d[:3] / homog_pts4d[3]).T  # shape (N,3)
            print('3d pts', points3D.shape)
            
            forward = R.T @ np.float32([0,0,1])
            origin = (-R.T @ t).T[0]
            origins.append(origin)
            frw = (origin + forward*0.2)
            origins.append(frw)
            
            forward = R0.T @ np.float32([0,0,1])
            origin = (-R0.T @ t0).T[0]
            origins.append(origin)
            frw = (origin + forward*0.2)
            origins.append(frw)
            
            show_points3D_2(np.float32(origins), points3D)
            #show_points3D(points3D)
            #exit()
        
        #print(corners1)        
        # fck this
        if len(corners1.shape) == 2:
            print('!!!???!!!??')
            corners1 = np.expand_dims(corners1, axis=1)
            corners2 = np.expand_dims(corners2, axis=1)
        #print(corners1)
        
        #print(objp.shape)
        #print(points3D.squeeze().shape)
        # fck this also
        pts3Dsqz = points3D.squeeze()
        
        print('OBJP', pts3Dsqz.shape)
        print('CORNERS', corners1.shape)
        assert len(pts3Dsqz.shape)==2 and len(corners1.shape)==3, "wrong input for solvePNP"
        #ret, rvec1, tvec1 = cv2.solvePnP(pts3Dsqz, corners1, newcameramtx, dist)
        #ret, rvec2, tvec2 = cv2.solvePnP(pts3Dsqz, corners2, newcameramtx, dist)
        ret, rvec1, tvec1, inl1 = cv2.solvePnPRansac(pts3Dsqz, corners1, newcameramtx, dist)
        ret, rvec2, tvec2, inl2 = cv2.solvePnPRansac(pts3Dsqz, corners2, newcameramtx, dist)
        
        R1, _ = cv2.Rodrigues(rvec1)
        R2, _ = cv2.Rodrigues(rvec2)
        
        orig1, fwd1 = get_cam_pos_and_dir(R1, tvec1)
        orig2, fwd2 = get_cam_pos_and_dir(R2, tvec2)
        
        rec_origins.append(orig1)
        rec_origins.append(orig1 + fwd1*0.2)
        rec_origins.append(orig2)
        rec_origins.append(orig2 + fwd2*0.2)
        
        #show_points3D_2(np.float32(rec_origins), points3D)
        
        #print(origins)
        #print(rec_origins)
        
        #exit()
        
    show_points3D_2(np.float32(rec_origins), points3D)    
        
    #print(origins)
    #show_points3D(np.float32(translations), np.zeros((3,1)))
    #show_points3D(np.float32(origins), np.zeros((3,1)))
    #show_points3D_2(np.float32(origins), objp)
    
def main3():        
    images_path = "./data/truck2"
    
    mtx = np.array([[1.61976266e+03, 0.00000000e+00, 1.03544002e+03],
            [0.00000000e+00, 1.62089834e+03, 7.72818235e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    dist = np.array([[ 0.14538323, -0.31737218, -0.00070913, -0.00119763, 0.12225343]])
    rvecs, tvecs = None, None
    newcameramtx = np.array([[1.64350977e+03, 0.00000000e+00, 1.03233371e+03],
                                [0.00000000e+00, 1.65042053e+03, 7.71463391e+02],
                                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    
    roi = (0, 0, 2080, 1560)
    
    
    images = glob.glob(os.path.join(images_path,'*.jpg'))

    
    translations = []
    #origins = []
    rec_origins = []
    points3D = None
    points3Dinit = None
    
    #point_clouds = []   # list of (points3D, keypoints, descriptors)

    for i in range(0, len(images)-1):
    #for i in range(len(images)-1):
    #for i in range(3):
        origins = []
        print(images[i], images[i+1])
        img1 = get_and_process_img(images[i], mtx, dist, newcameramtx)
        img2 = get_and_process_img(images[i+1], mtx, dist, newcameramtx)
        #cv2.imshow('im1', img1)
        #cv2.imshow('im2', img2)
        
        # find feature points...
        
        if points3D is None:
        #if len(point_clouds) == 0:
            #detector = cv2.SIFT_create(contrastThreshold=0.04, edgeThreshold=10)
            #detector = cv2.SIFT_create(contrastThreshold=0.04*2, edgeThreshold=10*0.5)          # or ORB_create(), AKAZE_create(), etc.        
            detector = cv2.SIFT_create()
            kpts1, desc1 = detector.detectAndCompute(img1, None)
            kpts2, desc2 = detector.detectAndCompute(img2, None)            
            matcher = cv2.FlannBasedMatcher()
            raw_matches = matcher.knnMatch(desc1, desc2, k=2)            
            good = []
            for m,n in raw_matches:
                if m.distance < 0.7 * n.distance:
                    good.append(m)
                    
            matchesImg = cv2.drawMatches(img1, kpts1, img2, kpts2, good, None)
            cv2.imshow('kp', dnsz(matchesImg))    
            cv2.waitKey() 
            
            corners1 = np.float32([kpts1[m.queryIdx].pt for m in good])
            corners2 = np.float32([kpts2[m.trainIdx].pt for m in good])
            used_kpts = np.array([kpts2[m.trainIdx] for m in good])
            used_descs = np.array([desc2[m.trainIdx] for m in good])
        
            #print(corners1)
            corners1 = corners1.squeeze()
            corners2 = corners2.squeeze()
            assert len(corners1.shape) == 2 and corners1.shape[1] == 2, "wrong input coordinates for pose recovery"
            E, mask = cv2.findEssentialMat(corners1, corners2, newcameramtx, cv2.RANSAC, 0.999, 1.0)
            res, R, t, mask_pose = cv2.recoverPose(E, corners1, corners2, newcameramtx, mask=mask)    
            #print(corners1)
            print('all pts', len(corners1))
            print('recover', res)
            #print('mask', mask)
            
            R0 = np.eye(3)
            t0 = np.zeros((3,1))
            P1 = np.hstack((R0, t0))
            P2 = np.hstack((R, t0+t))        

            P1 = newcameramtx @ P1
            P2 = newcameramtx @ P2

            inliers = mask_pose.ravel().astype(bool)
            #inliers[:] = True
            
            pts1_in = corners1[inliers].T
            pts2_in = corners2[inliers].T
            print(pts1_in.shape)

            homog_pts4d = cv2.triangulatePoints(P1, P2, pts1_in, pts2_in)        
            points3D = cv2.convertPointsFromHomogeneous(homog_pts4d.T)
            print('homog pts', homog_pts4d.shape)
            print('3d pts', points3D.shape)
            
            '''print(mask_pose)
            print(mask_pose.ravel())
            print('inliers\n', inliers)
            #print('kpts2\n', kpts2)
            print(len(kpts2), inliers.shape)
            used_kpts = used_kpts[inliers]
            print('used_kpts', used_kpts)
            exit()
            '''
            '''used_kpts  = np.array([ kpts2[i] for i,inl in enumerate(inliers) if inl ])
            used_descs = np.array([desc2[id] for id, inl in enumerate(inliers) if inl])
            used_kpts0  = np.array([ kpts1[i] for i,inl in enumerate(inliers) if inl ])
            used_descs0 = np.array([desc1[id] for id, inl in enumerate(inliers) if inl])
            corners1 = corners1[inliers]
            corners2 = corners2[inliers]
            '''
            
            used_kpts = used_kpts[inliers]
            used_descs = used_descs[inliers]
            corners1 = corners1[inliers]
            corners2 = corners2[inliers]            
            
            show_points3D(points3D)
            points3Dinit = points3D
            continue
            #exit()
        else:
            kpts2, desc2 = detector.detectAndCompute(img2, None)    # second keypoints for first img from pair
            matcher = cv2.FlannBasedMatcher()
            raw_matches = matcher.knnMatch(used_descs, desc2, k=2)            
            good = []
            for m,n in raw_matches:
                if m.distance < 0.6 * n.distance:
                    good.append(m)                    
            
            matchesImg = cv2.drawMatches(img1, used_kpts, img2, kpts2, good, None)
            cv2.imshow('kp', dnsz(matchesImg))    
            cv2.waitKey(100)
                    
            corners1 = np.float32([used_kpts[m.queryIdx].pt for m in good])
            corners2 = np.float32([kpts2[m.trainIdx].pt for m in good])
            used_kpts = np.array([kpts2[m.trainIdx] for m in good])
            used_descs = np.array([desc2[m.trainIdx] for m in good])
            points3D = np.array([points3D[m.queryIdx] for m in good])
        
        if len(corners1.shape) == 2:
            print('!!!???!!!??')
            corners1 = np.expand_dims(corners1, axis=1)
            corners2 = np.expand_dims(corners2, axis=1)
        pts3Dsqz = points3D.squeeze()
        
        print('OBJP', pts3Dsqz.shape)
        print('CORNERS', corners1.shape)
        assert len(pts3Dsqz.shape)==2 and len(corners1.shape)==3, "wrong input for solvePNP"
        pt_thr = 20
        if len(pts3Dsqz) < pt_thr or len(corners1) < pt_thr or len(corners2) < pt_thr:
            print('not enough features to recover pose')
            break
        #ret, rvec1, tvec1, inl1 = cv2.solvePnPRansac(pts3Dsqz, corners1, newcameramtx, dist)
        #ret, rvec2, tvec2, inl2 = cv2.solvePnPRansac(pts3Dsqz, corners2, newcameramtx, dist)
        
        #img_kp1 = cv2.drawKeypoints(img2, used_kpts, None)
        #cv2.imshow('kp1', dnsz(img_kp1))
        #show_points3D(points3D)
        
        ret, rvec1, tvec1, inl1 = cv2.solvePnPRansac(pts3Dsqz, corners1, newcameramtx, None)
        ret, rvec2, tvec2, inl2 = cv2.solvePnPRansac(pts3Dsqz, corners2, newcameramtx, None)
        print('num inliers', len(inl1) if inl1 is not None else 0, len(inl2) if inl2 is not None else 0)
        if inl1 is None or inl2 is None or max(len(inl1), len(inl2)) < 20:
            print('too little inliers')
            break
        
        
        R1, _ = cv2.Rodrigues(rvec1)
        R2, _ = cv2.Rodrigues(rvec2)
        
        orig1, fwd1 = get_cam_pos_and_dir(R1, tvec1)
        orig2, fwd2 = get_cam_pos_and_dir(R2, tvec2)
        
        rec_origins.append(orig1)
        rec_origins.append(orig1 + fwd1*0.2)
        rec_origins.append(orig2)
        rec_origins.append(orig2 + fwd2*0.2)
        
        # TODO: 1) full triangulation for new pair; 2) update 3d point cloud (or use latest and combine all)
        
        
    #show_points3D_2(np.float32(rec_origins), points3D)    
    show_points3D_2(np.float32(rec_origins), points3Dinit)    

def get_good_features(img1, img2, newcameramtx):
    detector = cv2.SIFT_create(contrastThreshold=0.04)
    #detector = cv2.SIFT_create(contrastThreshold=0.04, edgeThreshold=10)
    #detector = cv2.SIFT_create(contrastThreshold=0.04*2, edgeThreshold=10*2)          # or ORB_create(), AKAZE_create(), etc.                   
    kpts1, desc1 = detector.detectAndCompute(img1, None)
    kpts2, desc2 = detector.detectAndCompute(img2, None)            
    matcher = cv2.FlannBasedMatcher()
    raw_matches = matcher.knnMatch(desc1, desc2, k=2)            
    good = []
    for m,n in raw_matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
            
    #matchesImg = cv2.drawMatches(img1, kpts1, img2, kpts2, good, None)
    #cv2.imshow('kp', dnsz(matchesImg))    
    #cv2.waitKey(50) 
    
    corners1 = np.float32([kpts1[m.queryIdx].pt for m in good])
    corners2 = np.float32([kpts2[m.trainIdx].pt for m in good])
    used_kpts = np.array([kpts2[m.trainIdx] for m in good])
    used_descs = np.array([desc2[m.trainIdx] for m in good])

    corners1 = corners1.squeeze()
    corners2 = corners2.squeeze()
    assert len(corners1.shape) == 2 and corners1.shape[1] == 2, "wrong input coordinates for pose recovery"
    E, mask = cv2.findEssentialMat(corners1, corners2, newcameramtx, cv2.RANSAC, 0.999, 1.0)
    res, R, t, mask_pose = cv2.recoverPose(E, corners1, corners2, newcameramtx, mask=mask)    
    
    inliers = mask_pose.ravel().astype(bool)
        
    used_kpts = used_kpts[inliers]
    used_descs = used_descs[inliers]
    corners1 = corners1[inliers]
    corners2 = corners2[inliers]         
    
    colors = [img2[int(p[1]), int(p[0])] for p in corners2]
    
    return corners1, corners2, used_kpts, used_descs, colors
    
def reprojection_error(points3D, points2D, R, t, newcameramtx):
    rvec, _ = cv2.Rodrigues(R)    
    #rvec = R   # projectPoints works both with 3x3 and 3x1 option
    if points3D.shape[-1] == 3:
        points3D = points3D.reshape(-1, 1, 3)
    reproj_pts, jac = cv2.projectPoints(points3D, rvec, t, newcameramtx, None)
    reproj_pts = reproj_pts.reshape(-1, 2)
    points2D = points2D.reshape(-1, 2)    
    err = (np.linalg.norm(points2D-reproj_pts, axis = 0))
    rmse = np.sqrt(np.mean(err**2))
    return rmse

def do_pnp(pts3d_for_pnp, pts2d_for_pnp, K, iterations=500):    
    num_pts = len(pts3d_for_pnp)
    highest_inliers = 0
    error = 1e9
    R = np.zeros([3,1])
    tvec = np.zeros([3,1])
    for i in range(iterations):
        pt_idxs = np.random.choice(num_pts, 6, replace=False)
        pts3 = np.array([pts3d_for_pnp[pt_idxs[i]] for i in range(len(pt_idxs))])
        pts2 = np.array([pts2d_for_pnp[pt_idxs[i]] for i in range(len(pt_idxs))])        
        assert len(pts3.shape)==2 and len(pts2.shape)==3, "wrong input for solvePNP"
        res, rvec, tvec = cv2.solvePnP(pts3, pts2, K, distCoeffs=np.array([]), flags=cv2.SOLVEPNP_ITERATIVE)
        #res, rvec, tvec = cv2.solvePnP(pts3, pts2, K, distCoeffs=np.array([]), flags=cv2.SOLVEPNP_EPNP)
        if not res:
            continue
        R, _ = cv2.Rodrigues(rvec)        
        rmse = reprojection_error(pts3, pts2, R, tvec, K)
        if rmse < error:
            error = rmse            
            best_R = rvec
            best_tvec = tvec
             
    R = best_R
    tvec = best_tvec
    #print('rvec:', rvec,'\n\ntvec:', tvec)

    return R, tvec

def main4():        
    images_path = "./data/truck2"
    
    mtx = np.array([[1.61976266e+03, 0.00000000e+00, 1.03544002e+03],
            [0.00000000e+00, 1.62089834e+03, 7.72818235e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    dist = np.array([[ 0.14538323, -0.31737218, -0.00070913, -0.00119763, 0.12225343]])
    rvecs, tvecs = None, None
    newcameramtx = np.array([[1.64350977e+03, 0.00000000e+00, 1.03233371e+03],
                                [0.00000000e+00, 1.65042053e+03, 7.71463391e+02],
                                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    
    roi = (0, 0, 2080, 1560)
    
    
    images = glob.glob(os.path.join(images_path,'*.jpg'))

    
    translations = []
    rec_origins = []
    points3D = None
    points3Dinit = None
    
    PC, CLR, KP, DSC = 0, 1, 2, 3
    point_clouds = []   # list of (points3D, keypoints, descriptors)

    for i in range(0, len(images)-1):
    #for i in range(3):
        origins = []
        print(i, images[i], images[i+1])
        img1 = get_and_process_img(images[i], mtx, dist, newcameramtx)
        img2 = get_and_process_img(images[i+1], mtx, dist, newcameramtx)
        
        # find feature points...
        detector = cv2.SIFT_create(contrastThreshold=0.04)
        #if points3D is None:
        if len(point_clouds) == 0:
            
            kpts1, desc1 = detector.detectAndCompute(img1, None)
            kpts2, desc2 = detector.detectAndCompute(img2, None)          
            
            matcher = cv2.FlannBasedMatcher()
            #matcher = cv2.BFMatcher()
            raw_matches = matcher.knnMatch(desc1, desc2, k=2)            
            good = []
            for m,n in raw_matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)
                    
            #matchesImg = cv2.drawMatches(img1, kpts1, img2, kpts2, good, None)
            #cv2.imshow('kp', dnsz(matchesImg))    
            #cv2.waitKey() 
            
            corners1 = np.float32([kpts1[m.queryIdx].pt for m in good])
            corners2 = np.float32([kpts2[m.trainIdx].pt for m in good])
            used_kpts = np.array([kpts2[m.trainIdx] for m in good])
            used_descs = np.array([desc2[m.trainIdx] for m in good])
        
            corners1 = corners1.squeeze()
            corners2 = corners2.squeeze()
            assert len(corners1.shape) == 2 and corners1.shape[1] == 2, "wrong input coordinates for pose recovery"
            E, mask = cv2.findEssentialMat(corners1, corners2, newcameramtx, cv2.RANSAC, 0.999, 1.0)
            res, R, t, mask_pose = cv2.recoverPose(E, corners1, corners2, newcameramtx, mask=mask)    
            
            R0 = np.eye(3)
            t0 = np.zeros((3,1))
            P1 = np.hstack((R0, t0))
            P2 = np.hstack((R, t0+t))        

            P1 = newcameramtx @ P1
            P2 = newcameramtx @ P2

            inliers = mask_pose.ravel().astype(bool)
            
            pts1_in = corners1[inliers].T
            pts2_in = corners2[inliers].T
            
            homog_pts4d = cv2.triangulatePoints(P1, P2, pts1_in, pts2_in)        
            points3D = cv2.convertPointsFromHomogeneous(homog_pts4d.T)
                        
            rmse2 = reprojection_error(points3D, corners2[inliers], R, t, newcameramtx)
            rmse1 = reprojection_error(points3D, corners1[inliers], R0, t0, newcameramtx)            
            print('reproj err', rmse2, rmse1)
            #continue
            
            used_kpts = used_kpts[inliers]
            used_descs = used_descs[inliers]
            corners1 = corners1[inliers]
            corners2 = corners2[inliers]            
            
            #show_points3D(points3D)
            
            points3Dinit = points3D
            
            colors = [img2[int(p[1]), int(p[0])] for p in corners2]
            
            show_points3D_color(points3D, colors)
            
            point_clouds.append((points3D, colors, used_kpts, used_descs))
            continue
            
        else:
            kpts2, desc2 = detector.detectAndCompute(img2, None)    # second keypoints for first img from pair
            matcher = cv2.FlannBasedMatcher()
            #matcher = cv2.BFMatcher()
            raw_matches = matcher.knnMatch(point_clouds[-1][DSC], desc2, k=2)            
            good = []
            for m,n in raw_matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)                    
            
            matchesImg = cv2.drawMatches(img1, point_clouds[-1][KP], img2, kpts2, good, None)
            cv2.imshow('kp', dnsz(matchesImg))    
            cv2.waitKey(50)
                    
            corners1 = np.float32([point_clouds[-1][KP][m.queryIdx].pt for m in good])
            corners2 = np.float32([kpts2[m.trainIdx].pt for m in good])
            used_kpts = np.array([kpts2[m.trainIdx] for m in good])
            used_descs = np.array([desc2[m.trainIdx] for m in good])
            points3D = np.array([point_clouds[-1][PC][m.queryIdx] for m in good])
            
        
        if len(corners1.shape) == 2:
            print('!!!???!!!??')
            corners1 = np.expand_dims(corners1, axis=1)
            corners2 = np.expand_dims(corners2, axis=1)
        pts3Dsqz = points3D.squeeze()
        
        print('OBJP', pts3Dsqz.shape)
        print('CORNERS', corners1.shape)
        if len(pts3Dsqz) < 3 or len(corners1) < 3:
            print('too little points')
            continue
        assert len(pts3Dsqz.shape)==2 and len(corners1.shape)==3, "wrong input for solvePNP"
        pt_thr = 20
        if len(pts3Dsqz) < pt_thr or len(corners1) < pt_thr or len(corners2) < pt_thr:
            print('not enough features to recover pose')
            continue
        
        '''F, mask = cv2.findFundamentalMat(corners1, corners2, cv2.FM_RANSAC, 1.0, 0.999) # need to correctly sort pairs of points ??
        corners1 = corners1[mask.ravel() == 1]
        corners2 = corners2[mask.ravel() == 1]        
        pts3Dsqz = pts3Dsqz[mask.ravel() == 1]
        '''
        ret, rvec1, tvec1, inl1 = cv2.solvePnPRansac(pts3Dsqz, corners1, newcameramtx, None, flags=cv2.SOLVEPNP_ITERATIVE, reprojectionError=2.0, iterationsCount=1000)
        ret, rvec2, tvec2, inl2 = cv2.solvePnPRansac(pts3Dsqz, corners2, newcameramtx, None, flags=cv2.SOLVEPNP_ITERATIVE, reprojectionError=2.0, iterationsCount=1000)
        
        print('num inliers', len(inl1) if inl1 is not None else 0, len(inl2) if inl2 is not None else 0)
        if inl1 is None or inl2 is None or max(len(inl1), len(inl2)) < 20:
            print('too little inliers')
            continue
            
        rmse1 = reprojection_error(pts3Dsqz, corners1, rvec1, tvec1, newcameramtx)
        rmse2 = reprojection_error(pts3Dsqz, corners2, rvec2, tvec2, newcameramtx)                    
        print('approx reg reproj err', rmse1, rmse2)
        
        corn_inl1 = corners1[inl1].squeeze(axis=1)
        corn_inl2 = corners2[inl2].squeeze(axis=1)
        pts3D_inl1 = pts3Dsqz[inl1].squeeze(axis=1)
        pts3D_inl2 = pts3Dsqz[inl2].squeeze(axis=1)
            
        rvec1, tvec1 = cv2.solvePnPRefineLM(pts3D_inl1, corn_inl1, newcameramtx, None, rvec1, tvec1, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-6))
        rvec2, tvec2 = cv2.solvePnPRefineLM(pts3D_inl2, corn_inl2, newcameramtx, None, rvec2, tvec2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-6))
        #ret, rvec2, tvec2 = cv2.solvePnP(pts3D_inl2, corn_inl2, newcameramtx, None)
        #rvec2, tvec2 = do_pnp(pts3D_inl2, corn_inl2, newcameramtx)
        #rvec1, tvec1 = do_pnp(pts3Dsqz, corners1, newcameramtx)
        #rvec2, tvec2 = do_pnp(pts3Dsqz, corners2, newcameramtx)
        
        R1, _ = cv2.Rodrigues(rvec1)
        R2, _ = cv2.Rodrigues(rvec2)
        
        orig1, fwd1 = get_cam_pos_and_dir(R1, tvec1)
        orig2, fwd2 = get_cam_pos_and_dir(R2, tvec2)
        
        rec_origins.append(orig1)
        rec_origins.append(orig1 + fwd1*0.2)
        rec_origins.append(orig2)
        rec_origins.append(orig2 + fwd2*0.2)
        
        #P1 = np.hstack((R1.T, np.array([orig1]).T))   # .T ???
        #P2 = np.hstack((R2.T, np.array([orig2]).T))
        P1 = np.hstack((R1, tvec1))   # .T ???
        P2 = np.hstack((R2, tvec2))
        

        P1 = newcameramtx @ P1
        P2 = newcameramtx @ P2
        
        pts1, pts2, used_kpts, used_descs, colors = get_good_features(img1, img2, newcameramtx)

        homog_pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)     
        newpoints3D = cv2.convertPointsFromHomogeneous(homog_pts4d.T)
        
        rmse1 = reprojection_error(newpoints3D, pts1, R1, tvec1, newcameramtx)
        rmse2 = reprojection_error(newpoints3D, pts2, R2, tvec2, newcameramtx)                    
        print('big reg reproj err', rmse1, rmse2)
        
        #show_points3D_2(np.float32(rec_origins), newpoints3D)    
        
        point_clouds.append((newpoints3D, colors, used_kpts, used_descs))
        
        #if len(point_clouds) >= 2:
        #    show_points3D_2(point_clouds[-1][PC], point_clouds[-2][PC])
        
    #show_points3D_2(np.float32(rec_origins), points3Dinit)    
    merged_pcloud = point_clouds[0][PC]
    merged_colors = point_clouds[0][CLR]
    for cloud in point_clouds[1:]:        
        merged_pcloud = np.concatenate((merged_pcloud, cloud[PC]))
        merged_colors = np.concatenate((merged_colors, cloud[CLR]))
    
    #show_points3D_2(np.float32(rec_origins), merged_pcloud)
    show_points3D_2(np.float32(rec_origins), merged_pcloud, colors2=merged_colors)
    
if __name__ == "__main__":
    #main()
    #main2()
    #main3()
    main4()