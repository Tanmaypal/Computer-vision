import numpy as np
import cv2
from math import exp
import itertools
import math
##Create an array
'''Random_numbers=np.random.randint(1,20,20)
Random_numbers=Random_numbers.reshape(20,1)
print(Random_numbers)
def func(x):
    x=1/1+exp(x)
    return(x)
x1=func(6)
print(x1)

##
##Array2=np.array([func(i) for i in Random_numbers]
##print(Array2)
array=Array2.reshape(20,1)''
print(array)'''
##Instagram Filters
img_path="C:\\Users\\TANMAY\\Downloads\\lomo_1.jpg"
image=cv2.imread(img_path)
image1 = cv2.imread(img_path)
cv2.imshow("Image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

(B,G,R)=cv2.split(image)
'''cv2.imshow("image",R)
cv2.waitKey(0)
cv2.destroyAllWindows()'''
def curve_trans(x,s=0.01):
   return (1/(1+exp(((x-0.5)/s)*-1)))
channel_transform = np.vectorize(curve_trans)
result = channel_transform (R/255)*255
result=result.astype('uint8')
cv2.imshow("Result",result)
cv2.waitKey(0)
cv2.destroyAllWindows()
##
merge=cv2.merge([B,G,result])
cv2.imshow("Merge",np.hstack([image,merge]))
cv2.waitKey(0)
cv2.destroyAllWindows()
print(result.shape)
##
'''np.stack([image,merge])

print(result.dtype)

print(B.dtype)'''
##Filter over
(h,w,c)=image.shape
##
canvas=np.ones(image.shape[:2],dtype="uint8")*75
(centerX, centerY) = (canvas.shape[1] // 2, canvas.shape[0] // 2,)
white = (255, 255, 255)
cv2.circle(canvas,(centerX,centerY),(w//3),white,-1)
cv2.imshow("canvas",canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
#Gaussian blur
blurred =cv2.GaussianBlur(canvas, (193, 193),0)
cv2.imshow("Averaged", blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(blurred.shape)
(B,G,R)=cv2.split(image)
B=B.astype('float64')

##Convolution
'''''(B,G,R)=cv2.split(image)
B=B.astype('float64')
G=G.astype('float64')
R=R.astype('float64')
B1=cv2.multiply(B,blurred/255)
G1=cv2.multiply(G,blurred/255)
R1=cv2.multiply(R,blurred/255)
Merged=cv2.merge([B1,G1,R1])
cv2.imshow("New_image",Merged)
print(blurred/255.0)
cv2.waitKey(0)
cv2.destroyAllWindows()'''''
rows, cols = image.shape[:2]
# generating vignette mask using Gaussian kernels
kernel_x = cv2.getGaussianKernel(cols,200)
kernel_y = cv2.getGaussianKernel(rows,200)
kernel = kernel_y * kernel_x.T
mask = 255 * kernel / np.linalg.norm(kernel)
output = np.copy(image)
# applying the mask to each channel in the input image
##
def lomo_filter(image,s,vig_r):
    channel_transform = np.vectorize(curve_trans)
    result = channel_transform(R / 255,s) * 255
    result = result.astype('uint8')
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
kernel_x = cv2.getGaussianKernel(cols,200)
kernel_y = cv2.getGaussianKernel(rows,200)
kernel = kernel_y * kernel_x.T
mask = 255 * kernel / np.linalg.norm(kernel)
output = np.copy(image)

# applying the mask to each channel in the input image
for i in range(3):
   output[:,:,i] = output[:,:,i] * mask
cv2.imshow('Original', image)
cv2.imshow('Vignette', output)
cv2.waitKey(0)
cv2.destroyAllWindows()





