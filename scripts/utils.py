import cv2
import numpy as np

def HomogeneousCordinates(x_2d):
    x_homogenous = np.append(x_2d, np.ones((x_2d.shape[0],1)), axis=1)

    return x_homogenous

def skew(x):
    '''
    Inputs: a vector
    Outputs: skew symmetric matrix
    '''
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

def estimateEpipole(F):
    '''
    Inputs:
        Fundamental Matrix F
    Outputs:
        Epipole
    Since epilines should pass through the epipole estimate epipole using the formula: F @ e = 0
    '''
    _,_,V = np.linalg.svd(F)
    e = V[-1,:]
    e /= e[-1]
    return e


def draw_epipolar_lines(F, pts1, pts2, image1, image2):
    """
    Draws epipolar lines on both images given a fundamental matrix and corresponding points
    """

    # Compute epipole in second image
    e2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    e2 = e2.reshape(-1,3)

    # Compute epipole in first image
    e1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    e1 = e1.reshape(-1,3)

    # Draw lines on first image
    for epipolar_line in e1:
        x0, y0 = map(int, [0, -epipolar_line[2]/epipolar_line[1]])
        x1, y1 = map(int, [image1.shape[1], -(epipolar_line[2] + epipolar_line[0] * image1.shape[1])/epipolar_line[1]])
        cv2.line(image1, (x0, y0), (x1, y1), (0, 255, 0), 1)

    # Draw lines on second image
    for epipolar_line in e2:
        x0, y0 = map(int, [0, -epipolar_line[2]/epipolar_line[1]])
        x1, y1 = map(int, [image2.shape[1], -(epipolar_line[2] + epipolar_line[0] * image2.shape[1])/epipolar_line[1]])
        cv2.line(image2, (x0, y0), (x1, y1), (0, 255, 0), 1)

    # # Display the images with epipolar lines
    # cv2.imshow("Image 1 with epipolar lines", img1)
    # cv2.imshow("Image 2 with epipolar lines", img2)
    # cv2.waitKey(0)
    