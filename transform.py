# import the necessary packages
import numpy as np
import cv2

def order_points(pts):
    # points order: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4,2), dtype = "float32")

    # top-left point will have the smallest sum
    # bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # top-right point have the smallest difference
    # bottom-left point have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistend order of the points and unpack them incividually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of new image: maximun distance between br and bl x-coordinates or tr and tl x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of new image: maximun distance between tr and br y-coordinates or tl and bl y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # construct the set of destination points to obtain the "bird eye view" of the image
    dst = np.array([[0,0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight-1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped
