 import cv2
 import numpy as np
image=cv2.imread("C:\\Users\\TANMAY\\Downloads\\np2.jpg")
dim = (600, int(image.shape[0] * (600/image.shape[1])))
#image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
cv2.imshow("plate",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
##
##Gray
Gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray",Gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
##Gaussian Blurr
Blurr=cv2.GaussianBlur(Gray,(3,3),0)
cv2.imshow("Blurr",Blurr)
cv2.waitKey(0)
cv2.destroyAllWindows()
##Canny
Canny=cv2.Canny(Blurr,40,80)
cv2.imshow("CAnny",Canny)
cv2.waitKey(0)
cv2.destroyAllWindows()
##
(a,cnts,__) = cv2.findContours(Canny.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
print("I count {} coins in this image".format(len(cnts)))
Recipt = image.copy()
cv2.drawContours(Recipt, cnts, -1, (0, 0, 255), 2)
cv2.imshow("contours", Recipt)
cv2.waitKey(0)
##Sorting Contours
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
for i in cnts:
    peri1=cv2.arcLength(i,True)
    Approx1=cv2.approxPolyDP(i,0.02*peri1,True)
    if len(Approx1)==4:
        break


#print(peri1)
#print(Approx1)

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect
def four_point_transform(image, pts):
   # obtain a consistent order of the points and unpack them
   # individually
   rect = order_points(pts)
   #print(rect)
   (tl, tr, br, bl) = rect
   # compute the width of the new image, which will be the
   # maximum distance between bottom-right and bottom-left
   # x-coordiates or the top-right and top-left x-coordinates
   widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
   widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
   maxWidth = max(int(widthA), int(widthB))
   # compute the height of the new image, which will be the
   # maximum distance between the top-right and bottom-right
   # y-coordinates or the top-left and bottom-left y-coordinates
   heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
   heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
   maxHeight = max(int(heightA), int(heightB))
   # now that we have the dimensions of the new image, construct
   # the set of destination points to obtain a "birds eye view",
   # (i.e. top-down view) of the image, again specifying points
   # in the top-left, top-right, bottom-right, and bottom-left
   # order
   dst = np.array([
       [0, 0],
       [maxWidth - 1, 0],
       [maxWidth - 1, maxHeight - 1],
       [0, maxHeight - 1]], dtype="float32")
   # compute the perspective transform matrix and then apply it
   M = cv2.getPerspectiveTransform(rect, dst)
   warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
   # return the warped image
   return warped
##
pts = np.array([Approx1], dtype = "float32")
pts=pts.reshape(4,2)
warped = four_point_transform(image, pts)
cv2.imshow("Scanned1", warped)
cv2.waitKey(0)
##number_plate=cv2.imwrite("C:\\Users\\TANMAY\\Computer vision.jpg",warped)
##

warped = cv2.resize(warped,(128,81))
