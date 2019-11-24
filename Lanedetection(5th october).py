import numpy as np
import cv2
image=cv2.imread("C:\\Users\\TANMAY\\Downloads\\lanes3.png")
def color_filter(image):
   #convert to HLS to mask based on HLS
   #image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
   lower = np.array([0,190,0])
   upper = np.array([255,255,255])
   yellower = np.array([10,0,90])
   yelupper = np.array([50,255,255])
   yellowmask = cv2.inRange(hls, yellower, yelupper)
   whitemask = cv2.inRange(hls, lower, upper)
   mask = cv2.bitwise_or(yellowmask, whitemask)
   masked = cv2.bitwise_and(image, image, mask = mask)
   return masked


masked= color_filter(image)
cv2.imshow("Lane",masked)
#image1=cv2.imread("C:\\Users\\TANMAY\\Downloads\\lanes3.jpg")

cv2.waitKey(0)
cv2.destroyAllWindows()
##
def roi(img):
   x = int(img.shape[1])
   y = int(img.shape[0])
   shape = np.array([[int(0), int(y)], [int(x), int(y)], [int(0.55*x), int(0.6*y)], [int(0.45*x), int(0.6*y)]])
   #define a numpy array with the dimensions of img, but comprised of zeros
   mask = np.zeros_like(img)
   #Uses 3 channels or 1 channel for color depending on input image
   if len(img.shape) > 2:
       channel_count = img.shape[2]
       ignore_mask_color = (255,) * channel_count
   else:
       ignore_mask_color = 255
   #creates a polygon with the mask color
   cv2.fillPoly(mask, np.int32([shape]), ignore_mask_color)
   #returns the image only where the mask pixels are not zero
   masked_image = cv2.bitwise_and(img, mask)
   return masked_image
img=masked
masked_image=roi(masked)
RoI=masked_image
cv2.imshow("Output1",RoI)
cv2.waitKey(0)
cv2.destroyAllWindows()
##
canny=cv2.Canny(RoI,30,150)
cv2.imshow("canny",canny)
cv2.waitKey(0)
cv2.destroyAllWindows()
lines = cv2.HoughLinesP(canny, 1, np.pi/180, 20, minLineLength=1, maxLineGap=210)
# Draw lines on the image
for line in lines:
   x1, y1, x2, y2 = line[0]
   cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
# Show result
cv2.imshow("Result Image2",image)
cv2.waitKey(0)
