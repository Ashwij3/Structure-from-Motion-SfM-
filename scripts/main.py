import cv2
import numpy as np
from feature import *
from utils import *
from camera_pose import *


def main():
    debug = False
    K = np.asarray([
        [350, 0, 480],
        [0, 350, 270],
        [0, 0, 1]])
    
    img1=cv2.imread("/home/ak47/Major_Proj/SFM/Structure-from-Motion-SfM-/data/image0000001.jpg")
    img2=cv2.imread("/home/ak47/Major_Proj/SFM/Structure-from-Motion-SfM-/data/image0000002.jpg")

    x1, x2, _ = feature_extraction(queryImage = img1, trainImage = img2, ratioThresh = 0.4)


    # x1_homogeneous = cv2.convertPointsToHomogeneous(x1)
    # x2_homogeneous = cv2.convertPointsToHomogeneous(x2)
    # best_F, best_inliers_idx = cv2.findFundamentalMat(x1_homogeneous, x2_homogeneous, cv2.FM_RANSAC)
    best_F, best_inliers_idx = estimate_fundamental_ransac(x1,x2,10000,0.1)

    E = estimate_essential_matrix(K, best_F)


    x1 = x1[best_inliers_idx]
    x2 = x2[best_inliers_idx]

    if debug:
        image1 = img1.copy()
        image2 = img2.copy()
        for i in range(x1.shape[0]):
            cv2.circle(image1,(int(x1[i][0]),int(x1[i][1])),4,(255,0,0),-1)
            cv2.circle(image2,(int(x2[i][0]),int(x2[i][1])),4,(0,255,0),-1)
        draw_epipolar_lines(best_F, x1, x1, image1, image2)
        cv2.imshow("img1",image1)
        cv2.imshow("img2",image2)
        cv2.waitKey(0)
    

        

    camera_poses = extract_camera_pose(E)
    

    X_3D, best_pose, i = DisambiguateCameraPose(camera_poses, x1, x2, K)

    C1,R1 = np.zeros((3,1)), np.eye(3)
    X = NonlinearTriangulation(X_3D, x1, x2,[C1,R1],best_pose,K)

    best_C, best_R, max_inliers = PnP_RANSAC(X,x2,K,1000,20000)


    optimised_C, optimised_R = NonlinearPnP(X, x2, [best_C, best_R], K)




if __name__ == "__main__":
    main()