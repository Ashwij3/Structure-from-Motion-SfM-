import cv2
import numpy as np
from utils import skew


def ExtractCameraPose(E):
    """
    Find four conﬁgurations of rotation and camera center from E

    Parameters
    ----------
    E : ndarray of shape (3, 3)
        Essential Matrix

    Returns
    -------
    pose_list : list 
        4 camera conﬁgurations
    """
    pose_list = []
    W = np.asarray([
            [0,-1,0],
            [1,0,0],
            [0,0,1]])
    u, s, vh = np.linalg.svd(E)
    C = np.array([u[:,-1],-u[:,-1]])
    R = np.array([u@W@vh, u@W.T@vh])
    
    for i in range(len(R)):
        if np.linalg.det(R[i])>0:
            pose_list.append([C[0],R[i]])
            pose_list.append([C[1],R[i]])
        else:
            pose_list.append([-C[0],-R[i]])
            pose_list.append([-C[1],-R[i]])
    
    return pose_list

def LinearTriangulation(C_R1, C_R2, x1_2D, x2_2D, K):

    x1_2D=np.array(x1_2D)
    x2_2D=np.array(x2_2D)
    assert x1_2D.shape[0] == x2_2D.shape[0]

    x1_homogenous = np.append(x1_2D, np.ones((x1_2D.shape[0],1)), axis=1)
    x2_homogenous = np.append(x2_2D, np.ones((x2_2D.shape[0],1)), axis=1)

    X =[]
    C1,R1 = C_R1
    C2,R2 = C_R2

    T1 = K @ np.hstack((R1, C1.reshape((-1,1))))
    T2 = K @ np.hstack((R2, C2.reshape((-1,1))))
    
    for point1,point2 in zip(x1_homogenous,x2_homogenous):

        A = [[skew(point1)@T1],
             [skew(point2)@T2]]
        
        AX = np.array(A).reshape((-1,4))

        _,_,V = np.linalg.svd(AX)
        Xi = V[:,-1]/V[-1,-1]
        X.append(Xi)

    X = np.array(X)

    return X

def DisambiguateCameraPose(pose_list,p1,p2,K):

    p1=np.array(p1)
    p2=np.array(p2)
    assert p1.shape[0] == p2.shape[0]

    best_pose = []
    max_inlier_count = 0
    optimal_X = []
    
    C1,R1 = np.zeros((3,1)), np.eye(3)
    for pose in pose_list:
        C2,R2 = pose
        X = LinearTriangulation([C1,R1],[C2,R2],p1,p2,K)

        chirality = R2[:,2] @ (X[:,:3] - C2).T
        pos_Z = X[:,2]>0

        inliers = (chirality>0  & pos_Z).sum()
        if(inliers>max_inlier_count):
            max_inlier_count = inliers
            best_pose.append([C2,R2])
            optimal_X = X

    return optimal_X, best_pose, max_inlier_count

def NonlinearTriangulation(X_3D,p1,p2):
    
    def loss_fnc(params):
        pass