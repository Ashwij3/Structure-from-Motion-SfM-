import cv2
import numpy as np
from feature import *
from camera_pose import *


def main():

    K = np.asarray([
        [350, 0, 480],
        [0, 350, 270],
        [0, 0, 1]])
    
    img1=cv2.imread("/home/ak47/Major_Proj/SFM/SFM/data/image0000001.jpg")
    img2=cv2.imread("/home/ak47/Major_Proj/SFM/SFM/data/image0000002.jpg")

    sift = cv2.SIFT_create()
    k1,d1=sift.detectAndCompute(img1,None)
    k2,d2=sift.detectAndCompute(img2,None)

    loc1, loc2 = [], []
    for i in range(0,len(k1)):
        loc1.append([k1[i].pt[0],k1[i].pt[1]])

    for i in range(0,len(k2)):
        loc2.append([k2[i].pt[0],k2[i].pt[1]])

    x1, x2, _ = MatchSIFT(loc1,d1,loc2,d2)

    F, inliers = EstimateF_RANSAC(x1,x2,100,0.2)
    print(len(inliers))
    F_cv, mask = cv2.findFundamentalMat(x1,x2,cv2.FM_RANSAC)
    print(len(np.where(mask ==1)[0]))
    # draw_epipolar_lines(F, x1, x2, img1, img2)
    # draw_epipolar_lines(F_cv, x1, x2, img1, img2)
    print(F)
    print("+++++++++++++")
    print(F_cv)

    E = EstimateE(K,F)
    pose_list = ExtractCameraPose(E)
    optimal_X, best_pose, max_inlier_count = DisambiguateCameraPose(pose_list, x1, x2, K)


if __name__ == "__main__":
    main()


