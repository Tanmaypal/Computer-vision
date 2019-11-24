import numpy as np
import cv2
img_path="C:\\Users\\TANMAY\\Downloads\\lenna.png"
image=cv2.imread(img_path)
cv2.imshow("Image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
Gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray",Gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
##
blurred =cv2.GaussianBlur(Gray, (7,7),0)
cv2.imshow("Gaussian", blurred)
cv2.waitKey(0)
cv2.waitKey(0)
cv2.destroyAllWindows()
##
edges = cv2.Laplacian(blurred, cv2.CV_8U, ksize=5)
ret, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("Edges", edges)
cv2.waitKey(0)
ds_factor = 4
img_small = cv2.resize(image, None, fx=1.0/ds_factor, fy=1.0/ds_factor,interpolation=cv2.INTER_AREA)
cv2.imshow("Img_small",img_small)
cv2.waitKey(0)
cv2.destroyAllWindows()

num_repetitions = 15
for i in range(num_repetitions):
    img_small = cv2.bilateralFilter(img_small, 5, 5, 7)

img_output = cv2.resize(img_small, None, fx=ds_factor, fy=ds_factor,interpolation=cv2.INTER_LINEAR)
cv2.imshow("im1",img_output)
cv2.waitKey(0)

dst = np.zeros(Gray.shape)
dst = cv2.bitwise_and(img_output, img_output, mask=mask)
cv2.imshow("Cartoon", dst)
cv2.waitKey(0)
###Function of cartoonize
def Cartoonize(image,num_iters):
    num_repetitions = num_iters
    for i in range(num_repetitions):
        img_small = cv2.bilateralFilter(img_small, 5, 5, 7)

img_output = cv2.resize(img_small, None, fx=ds_factor, fy=ds_factor,interpolation=cv2.INTER_LINEAR)
cv2.imshow("im1",img_output)
cv2.waitKey(0)

dst = np.zeros(Gray.shape)
dst = cv2.bitwise_and(img_output, img_output, mask=mask)
cv2.imshow("Cartoon", dst)
cv2.waitKey(0)
