
import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path="C:\\Users\\TANMAY\\Desktop\\pisa1.jpg"
image=cv2.imread(img_path)
img1=abs(255-image)
cv2.imshow("Img2",img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
##Resize of an image
output=cv2.resize(image,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
cv2.imshow("Output",output)
cv2.waitKey(0)
cv2.destroyAllWindows()