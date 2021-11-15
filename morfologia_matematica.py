# -*- coding: utf-8 -*-
"""
Created on Mon May 13 18:54:25 2019

@author: jeans
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Image4.jpeg',0)
#img = cv2.imread('9.png',0)
kernel = np.ones((8,8),np.uint8)
#erosion = cv2.erode(img,kernel,iterations = 1)
#dilation = cv2.dilate(img,kernel,iterations = 1)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
#closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
#gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

#calculo = erosion
#calculo = dilation
calculo = opening
#calculo = closing
#calculo = gradient

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(calculo),plt.title('Calculo')
plt.xticks([]), plt.yticks([])
plt.show()