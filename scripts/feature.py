import cv2
from typing import Tuple
import numpy as np
from utils import *




def feature_extraction(queryImage: np.ndarray, trainImage: np.ndarray, ratioThresh: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts SIFT features and matches them between two images using brute-force matching.

    Args:
        queryImage (numpy.ndarray): The image to extract SIFT features from.
        trainImage (numpy.ndarray): The image to match SIFT features against.
        ratioThresh (numpy.ndarray): The ratio test threshold for selecting good matches.

    Returns:
        A tuple containing:
        - x1: A numpy array of shape (N, 2) containing the (x, y) coordinates of matched keypoints in the query image.
        - x2: A numpy array of shape (N, 2) containing the (x, y) coordinates of matched keypoints in the train image.
        - ind1: A numpy array of shape (N,) containing the indices of matched keypoints in the query image.

    Raises:
        ValueError: If the shape of x1 and x2 do not match.
    """
    # Create a SIFT object
    sift = cv2.SIFT_create()

    # Detect and compute SIFT features for both images
    k1, des1 = sift.detectAndCompute(queryImage, None)
    k2, des2 = sift.detectAndCompute(trainImage, None)

    # Perform brute-force matching with ratio test
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # Select good matches based on the ratio test
    x1, x2, ind1 = [], [], []
    for _, (m, n) in enumerate(matches):
        if m.distance < ratioThresh * n.distance:
            x1.append([k1[m.queryIdx].pt[0], k1[m.queryIdx].pt[1]])
            x2.append([k2[m.trainIdx].pt[0], k2[m.trainIdx].pt[1]])
            ind1.append(m.queryIdx)

    # Convert the matched keypoints to numpy arrays
    x1 = np.array(x1)   
    x2 = np.array(x2)
    ind1 = np.array(ind1)

    # Check if x1 and x2 have the same shape
    if x1.shape != x2.shape:
        raise ValueError("x1 and x2 have different shapes")

    return x1, x2, ind1



def estimate_fundamental_matrix(x1_2d: np.ndarray, x2_2d: np.ndarray) -> np.ndarray:
    """
    Estimates the fundamental matrix given two sets of corresponding 2D points.

    Args:
        x1_2d (numpy.ndarray): A numpy array of shape (N, 2) containing the (x, y) coordinates of points in the first image.
        x2_2d (numpy.ndarray): A numpy array of shape (N, 2) containing the (x, y) coordinates of points in the second image.

    Returns:
        F: A 3x3 numpy array representing the estimated fundamental matrix.

    Raises:
        ValueError: If the shapes of x1_2d and x2_2d do not match or the number of points is less than 8.
    """
    # Check the shapes of x1_2d and x2_2d
    if x1_2d.shape != x2_2d.shape:
        raise ValueError("x1_2d and x2_2d have different shapes")

    # Check the number of points
    n = x1_2d.shape[0]
    if n < 8:
        raise ValueError("At least 8 corresponding points are required")

    # Create the A matrix
    x1, y1 = x1_2d[:, 0], x1_2d[:, 1]
    x2, y2 = x2_2d[:, 0], x2_2d[:, 1]
    ones = np.ones((n,))
    A = np.column_stack((x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, ones))

    # Compute the fundamental matrix using SVD
    _, _, v = np.linalg.svd(A)
    F = v[-1, :].reshape((3, 3))

    # Enforce the rank-2 constraint on F
    u, s, v = np.linalg.svd(F)
    s[-1] = 0
    F = u @ np.diag(s) @ v

    return F



def estimate_fundamental_ransac(x1_2d: np.ndarray, x2_2d: np.ndarray, ransac_iter: int, ransac_threshold: float) -> tuple:
    """
    Estimate fundamental matrix using RANSAC algorithm.

    Args:
    x1_2d (numpy.ndarray): Array of shape (n, 2) representing 2D coordinates of points in image 1.
    x2_2d (numpy.ndarray): Array of shape (n, 2) representing 2D coordinates of points in image 2.
    ransac_iter (int): Number of RANSAC iterations to perform.
    ransac_threshold (float): Threshold for RANSAC algorithm.

    Returns:
    A tuple containing:
        - best_F: Optimal fundamental matrix using RANSAC
        - best_inliers_idx: Indices of inliers.
        
    """

    # Check the shapes of x1_2d and x2_2d
    if x1_2d.shape != x2_2d.shape:
        raise ValueError("x1_2d and x2_2d have different shapes")

    # Get the number of points
    n = x1_2d.shape[0] 

    # Convert points to homogeneous coordinates
    x1_homogeneous = HomogeneousCordinates(x1_2d)
    x2_homogeneous = HomogeneousCordinates(x2_2d)

    # Initialize best inliers, best fundamental matrix
    best_inliers_idx = []
    best_F = np.eye(3)

    # Perform RANSAC iterations
    for _ in range(ransac_iter):

        # Select 8 random points
        rand_points_idx = np.random.choice(n, size=8, replace=False)

         # Estimate fundamental matrix using the random points
        F = estimate_fundamental_matrix(x1_homogeneous[rand_points_idx], x2_homogeneous[rand_points_idx])

        # Compute epipolar lines and errors
        lines2 = F @ x1_homogeneous.T
        error = abs(np.sum(np.multiply(x2_homogeneous, lines2.T),axis=1))

        # Identify inliers based on threshold
        inliers_idx = np.where(error < ransac_threshold)[0]

        if len(inliers_idx) > len(best_inliers_idx):
            best_inliers_idx = inliers_idx
    
    # Compute optimal fundamental matrix using the best inliers
    best_F = estimate_fundamental_matrix(x1_homogeneous[best_inliers_idx],x2_homogeneous[best_inliers_idx])

    return best_F, best_inliers_idx
    


def estimate_essential_matrix(K: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    Computes the essential matrix from the camera calibration matrix K and the fundamental matrix F.

    Args:
        K (np.ndarray): Camera calibration matrix, 3x3 numpy array.
        F (np.ndarray): Fundamental matrix, 3x3 numpy array.

    Returns:
        E: Essential matrix, 3x3 numpy array.

    """
    E = K.T @ F @ K
    u, s, v = np.linalg.svd(E)
   
    s = [1,1,0]
    E = u@np.diag(s)@v

    return E

