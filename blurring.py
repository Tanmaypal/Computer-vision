import cv2
import numpy as np
from matplotlib import pyplot as plt
image=cv2.imread("C:\\Users\\TANMAY\\Downloads\\coins.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)

###Resize
r = 300.0 / image.shape[1]
dim = (300, int(image.shape[0] * r))
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
cv2.imshow("Resized (Width)", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

##Blurred
blurred = np.hstack([
   cv2.blur(resized, (3, 3)),
   cv2.blur(resized, (5, 5)),
   cv2.blur(resized, (13, 13))])
cv2.imshow("Averaged", blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()
##gray

##Gaussian Blurred
blurred = np.hstack([
   cv2.GaussianBlur(resized, (3, 3),0),
   cv2.GaussianBlur(resized, (5, 5),0),
   cv2.GaussianBlur(resized, (7, 7),0)])
cv2.imshow("Gaussian", blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()
##Bilateral Filter
cv2.bilateralFilter(resized, 5, 21, 21)
cv2.imshow("Bilateral",resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(image.shape)
####Threshold
(T, thresh) = cv2.threshold(blurred, 245, 255, cv2.THRESH_BINARY)
cv2.imshow("Threshold Binary", thresh)##values above Threshold is given a standard value
(T, threshInv) = cv2.threshold(blurred, 245, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("Threshold Binary Inverse", threshInv)
cv2.waitKey(0)
cv2.destroyAllWindows()
#####Adaptive Threshold is always applied on Gray
thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
cv2.imshow("Mean Thresh", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
###Adaptive mean
thresh = cv2.adaptiveThreshold(gray, 255,
cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
cv2.imshow("Gaussian Thresh", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
###
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original", image)
lap = cv2.Laplacian(gray, cv2.CV_64F)##Laplacian is used to detect edges as inward and outward edges
lap = np.uint8(np.absolute(lap))
cv2.imshow("Laplacian", lap)
cv2.waitKey(0)
cv2.destroyAllWindows()
##Canny
canny = cv2.Canny(image, 100, 150)
cv2.imshow("CANNY",canny)
cv2.waitKey(0)
cv2.destroyAllWindows()
##canny uses multi stage algorithm to detect wide ranges of edges
def auto_canny(image, sigma=0.33):
   # compute the median of the single channel pixel intensities
   v = np.median(image)
   # apply automatic Canny edge detection using the computed median
   lower = int(max(0, (1.0 - sigma) * v))
   upper = int(min(255, (1.0 + sigma) * v))
   edged = cv2.Canny(image, lower, upper)
   # return the edged image
   return edged
(_,cnts,_)= cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL,
cv2.CHAIN_APPROX_SIMPLE)
print("I count {} coins in this image".format(len(cnts)))
coins = image.copy()
cv2.drawContours(coins, cnts, -1, (0, 0, 255), 2)
cv2.imshow("Coins", coins)
cv2.waitKey(0)
