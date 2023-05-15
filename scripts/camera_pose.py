import cv2
from typing import List, Tuple
import numpy as np
import scipy as sc
from scipy.spatial.transform import Rotation
from utils import skew


def extract_camera_pose(E: np.ndarray)-> List[List[np.ndarray]] :
    """
    Extract the possible camera poses from an essential matrix using the method of decomposition.

    Args:
    - E (np.ndarray): A 3x3 essential matrix

    Returns:
    - camera_poses: A list of two possible camera poses. Each pose is a list consisting of a 3x1 translation vector and a 3x3 rotation matrix.
    """

    # Initialize an empty list to store the camera poses
    camera_poses = []

    # Define the skew-symmetric matrix W
    W = np.asarray([[0,-1,0],[1,0,0],[0,0,1]])

    # Perform SVD on E
    U, _, V = np.linalg.svd(E)

    # Compute the two possible translations and rotations
    C = np.vstack((U[:,2], -U[:,2]))
    R = [U @ W @ V, U @ W.T @ V]
    
    for i in range(2):

        if np.linalg.det(R[i])>0:

            camera_poses.append([C[0],R[i]])
            camera_poses.append([C[1],R[i]])
        
        else:

            camera_poses.append([-C[0],-R[i]])
            camera_poses.append([-C[1],-R[i]])
    
    return camera_poses



def linear_triangulation(C_R1: Tuple[np.ndarray, np.ndarray], C_R2: Tuple[np.ndarray, np.ndarray], x1_2D: np.ndarray, 
                        x2_2D: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Linear triangulation of 3D points from two camera views and their 2D correspondences.
    
    Args:
    - C_R1: tuple of camera center and rotation matrix for first camera view
    - C_R2: tuple of camera center and rotation matrix for second camera view
    - x1_2D: 2D correspondences in image coordinates for first camera view
    - x2_2D: 2D correspondences in image coordinates for second camera view
    - K: camera intrinsic matrix
    
    Returns:
    - X: array of 3D points in homogeneous coordinates
    """
    # Check the shapes of x1_2d and x2_2d
    if x1_2D.shape != x2_2D.shape:
        raise ValueError("x1_2d and x2_2d have different shapes")

    x1_homogeneous = np.hstack((x1_2D, np.ones((x1_2D.shape[0], 1))))
    x2_homogeneous = np.hstack((x2_2D, np.ones((x2_2D.shape[0], 1))))

    X =[]
    C1,R1 = C_R1
    C2,R2 = C_R2

    t1 = -R1 @ C1
    t2 = -R2 @ C2

    T1 = K @ np.hstack((R1, t1.reshape((-1,1))))
    T2 = K @ np.hstack((R2, t2.reshape((-1,1))))
    
    for x1,x2 in zip(x1_homogeneous,x2_homogeneous):

        A = np.vstack((skew(x1) @ T1, skew(x2) @ T2))
        _, _, V = np.linalg.svd(A)
        Xi = V[-1, :] / V[-1, -1]
        X.append(Xi)

    X = np.array(X)

    return X



def DisambiguateCameraPose(pose_list,p1,p2,K):

    p1=np.array(p1)
    p2=np.array(p2)
    assert p1.shape[0] == p2.shape[0]


    max_inlier_count = 0
    optimal_X = []
    
    C1,R1 = np.zeros((3,1)), np.eye(3)
    for pose in pose_list:
        C2,R2 = pose
        X = linear_triangulation([C1,R1],[C2,R2],p1,p2,K)
        chirality = R2[:,2] @ (X[:,:3] - C2).T
        chirality_idx = np.where(chirality>0)[0]

        pos_Z_idx = np.where(X[:,2]>0)[0]
        
        inlier_idx = np.intersect1d(pos_Z_idx,chirality_idx)
        
        if(inlier_idx.shape[0]>max_inlier_count):
            max_inlier_count = inlier_idx.shape[0]
            max_inlier_idx = inlier_idx
            best_pose = [C2,R2]
            optimal_X = X

    return optimal_X, best_pose, max_inlier_idx

def NonlinearTriangulation(X_3D, x1_2D, x2_2D, C_R1, C_R2, K):
    
    C1, R1 = C_R1
    C2, R2 = C_R2

    t1 = -R1 @ C1
    t2 = -R2 @ C2

    T1 = K @ np.hstack((R1, t1.reshape((-1,1))))
    T2 = K @ np.hstack((R2, t2.reshape((-1,1))))
    
    def geometric_loss(params):
        X = params.reshape((-1,4))
        
        x1 = X @ T1.T
        x1 = x1/(x1[:,2]).reshape(-1,1)

        x2 = X @ T2.T
        x2 = x2/(x2[:,2]).reshape(-1,1)

        error = (np.square(x1[:,0] - x1_2D[:,0]) + np.square(x1[:,1] - x1_2D[:,1]) \
                +np.square(x2[:,0] - x2_2D[:,0]) + np.square(x2[:,1] - x2_2D[:,1])).sum()

        return error
    

    params = X_3D.reshape(-1)
    optimized = sc.optimize.least_squares(geometric_loss,  params, xtol=500, gtol=10, max_nfev=8)
    X_optimized = optimized.x.reshape((-1,4))

    return X_optimized

def linear_pnp(X_3D: np.ndarray, x_2D: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the camera pose using Linear Perspective-n-Point (PnP) algorithm.

    Args:
    X_3D (np.ndarray): An Nx3 array of 3D points in the world coordinates, where N is the number of points.
    x_2D (np.ndarray): An Nx2 array of corresponding 2D points in the image plane.
    K (np.ndarray): The camera intrinsic matrix.

    Returns:
    A tuple containing:
        -C: the camera center 
        -R: the rotation matrix
    """

    if x_2D.shape[0] != X_3D.shape[0]:
        raise ValueError("Not equal number of 2D-3D correspondence")

    num_points = X_3D.shape[0]
    if num_points < 6:
        raise ValueError("At least 6 corresponding points are required")

    zeros = np.zeros((num_points))
    ones = np.ones((num_points))

    A = np.zeros((num_points,12))

    X, Y, Z = X_3D[:,:3].T
    u, v = x_2D.T

    A1 = np.array([X, Y, Z, ones, zeros, zeros, zeros, zeros, -u*X, -u*Y, -u*Z, -u]).T
    A2 = np.array([zeros, zeros, zeros, zeros, X, Y, Z, ones,  -v*X, -v*Y, -v*Z, -v]).T

    A = np.vstack([A1,A2])

    _, _, v = np.linalg.svd(A)
    P = v[-1,:].reshape((3,4))

    R = np.linalg.inv(K) @ P[:,:3]
    uR,sR,vhR = np.linalg.svd(R)
    R = uR @ vhR

    t = np.linalg.inv(K) @ P[:,3] / sR[0]

    # Ensure a valid rotation matrix
    if np.linalg.det(R) < 0:
        R = -R
        t = -t 

    C = -R.T @ t

    return C,R


def PnP_RANSAC(X_3D: np.ndarray, x_2D: np.ndarray, K: np.ndarray, ransac_n_iter: int, ransac_thr: float) -> tuple:
    """
    Estimate camera pose using PnP with RANSAC.

    Args:
    - X_3D: 3D points in the world coordinate system (shape: [N, 3])
    - x_2D: 2D points in the image coordinate system (shape: [N, 2])
    - K: camera intrinsic matrix
    - ransac_n_iter: number of RANSAC iterations
    - ransac_thr: RANSAC inlier threshold

    Returns:
    - best_C: estimated camera center
    - best_R: estimated camera rotation matrix
    - best_inliers_idx: indices of the inlier correspondences

    Raises:
    - ValueError: If the number of 2D-3D correspondences is not equal
    """
    
    if x_2D.shape[0] != X_3D.shape[0]:
        raise ValueError("Not equal number of 2D-3D correspondence")

    num_points = X_3D.shape[0]

    best_inliers_idx = []
    best_C, best_R = None, None

    for _ in range(ransac_n_iter):

        rand_points_idx = np.random.choice(num_points, size=6, replace=False)
        C,R = linear_pnp(X_3D[rand_points_idx],x_2D[rand_points_idx],K)

        t = -R @ C

        P = K @ np.hstack((R, t.reshape((-1,1))))

        l = X_3D @ P.T 
        l /= l[:, [-1]]

        error = (l[:,0] - x_2D[:,0])**2 + (l[:,1] - x_2D[:,1])**2
        inlier_idx = np.where(abs(error)<ransac_thr)[0]

        if len(inlier_idx) > len(best_inliers_idx):
            best_inliers_idx = inlier_idx

        best_C,best_R = linear_pnp(X_3D[best_inliers_idx],x_2D[best_inliers_idx],K)

       

    
    return best_C, best_R, best_inliers_idx


def NonlinearPnP(X_3D, x_2D, camera_pose, K):
    C, R = camera_pose
    t = -R @ C
    q = Rotation.from_matrix(R).as_quat()
    params = np.hstack((q,t))
    params = params.flatten()

    def geometric_loss(params):
        q = params[:4]
        t = params[4:] 
        R = Rotation.from_quat(q).as_matrix()
        P = K @ np.hstack((R, t.reshape((-1,1))))
        l = X_3D @ P.T 
        l = l / l[:,[-1]]
        error = (l[:,0] - x_2D[:,0])**2 + (l[:,1] - x_2D[:,1])**2

        return error



    optimized = sc.optimize.least_squares(geometric_loss,  params, xtol=500, gtol=10, max_nfev=8)
    q = optimized.x[:4]
    R = Rotation.from_quat(q).as_matrix()

    t = optimized.x[4:]
    C = -R.T @ t 

    return C, R

    