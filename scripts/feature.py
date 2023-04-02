import cv2
import numpy as np



def MatchSIFT(loc1, des1, loc2, des2, ratio_thresh = 0.7):
    """
    Find the matches of SIFT features between two images
    
    Parameters
    ----------
    loc1 : ndarray of shape (n1, 2)
        Keypoint locations in image 1
    des1 : ndarray of shape (n1, 128)
        SIFT descriptors of the keypoints image 1
    loc2 : ndarray of shape (n2, 2)
        Keypoint locations in image 2
    des2 : ndarray of shape (n2, 128)
        SIFT descriptors of the keypoints image 2

    Returns
    -------
    x1 : ndarray of shape (m, 2)
        The matched keypoint locations in image 1
    x2 : ndarray of shape (m, 2)
        The matched keypoint locations in image 2
    ind1 : ndarray of shape (m,)
        The indices of x1 in loc1
    """

    x1,x2,ind1=[],[],[]

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2) #query,#train

    
    
    for _, (m, n) in enumerate(matches):
        if m.distance < ratio_thresh * n.distance:
            x1.append(loc1[m.queryIdx])
            x2.append(loc2[m.trainIdx])
            ind1.append(m.queryIdx)
        
    x1=np.array(x1)   
    x2=np.array(x2)
    ind1=np.array(ind1)

    return x1, x2, ind1



def EstimateF(x1_2D, x2_2D):
    """
    Estimate the Fundamental matrix, which is a rank 2 matrix with singular values
    (1, 1, 0) using 8 point algorithm

    Parameters
    ----------
    x1_2D : ndarray of shape (n, 2)
        Set of correspondences in the first image
    x2_2D : ndarray of shape (n, 2)
        Set of correspondences in the second image

    Returns
    -------
    F : ndarray of shape (3, 3)
        The fundamental matrix
    """
    
    x1_2D=np.array(x1_2D)
    x2_2D=np.array(x2_2D)
    assert x1_2D.shape[0] == x2_2D.shape[0]
    n = x1_2D.shape[0] 

    x1,y1 = x1_2D[:,0],x1_2D[:,1]
    x2,y2 = x2_2D[:,0],x2_2D[:,1]
    A = np.zeros((n,9))

    A[:,0] = x2*x1
    A[:,1] = x2*y1
    A[:,2] = x2
    A[:,3] = y2*x1
    A[:,4] = y2*y1
    A[:,5] = y2
    A[:,6] = x1
    A[:,7] = y1
    A[:,8] = 1

    u, s, vh = np.linalg.svd(A)

    F = vh[np.argmin(s),:]
    F = F.reshape((3,3))

    uF, sF, vhF = np.linalg.svd(F, full_matrices=False)

    sF[2] = 0
    F = uF@np.diag(sF)@vhF


    return F



def EstimateF_RANSAC(x1_2D, x2_2D, ransac_n_iter, ransac_thr):
    """
    Estimate the Fundamental matrix robustly using RANSAC

    Parameters
    ----------
    x1_2D : ndarray of shape (n, 2)
        Set of correspondences in the first image
    x2 : ndarray of shape (n, 2)
        Set of correspondences in the second image
    ransac_n_iter : int
        Number of RANSAC iterations
    ransac_thr : float
        Error threshold for RANSAC

    Returns
    -------
    F : ndarray of shape (3, 3)
        The Fundamnetal matrix
    inlier : ndarray of shape (k,)
        The inlier indices
    """
    
    x1_2D=np.array(x1_2D)
    x2_2D=np.array(x2_2D)
    assert x1_2D.shape[0] == x2_2D.shape[0]
    n = x1_2D.shape[0] 

    x1_homogenous = np.append(x1_2D, np.ones((n,1)), axis=1)
    x2_homogenous = np.append(x2_2D, np.ones((n,1)), axis=1)


    best_inliers = []

    for _ in range(ransac_n_iter):

        rand_points_idx = np.random.choice(n, size=8, replace=False)
        F = EstimateF(x1_homogenous[rand_points_idx],x2_homogenous[rand_points_idx])
        
        lines2 = F @ x1_homogenous.T
        error = abs(np.sum(np.multiply(x2_homogenous, lines2.T),axis=1))
        

        inliers = np.where(error < ransac_thr)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers  

    F_final = EstimateF(x1_homogenous[best_inliers], x2_homogenous[best_inliers]) 
    best_inliers = np.array(best_inliers)
    return F_final, best_inliers 


def EstimateE(K,F):
    """
    Estimate the Essential matrix which is a rank 2 matrix with singular values
    (1, 1, 0)

    Parameters
    ----------
    K : ndarray of shape (3, 3)
        Camera Calibration matrix
    F : ndarray of shape (3, 3)
        Fundamental Matrix

    Returns
    -------
    E : ndarray of shape (3, 3)
        The Essential matrix
    """
    E = K.T @ F @ K
    u, s, vh = np.linalg.svd(F)
   
    s = [1,1,0]
    E = u@np.diag(s)@vh

    return E
    
