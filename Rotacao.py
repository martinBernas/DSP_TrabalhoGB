
import numpy as np
import cv2

#Com o parametro '0' já lê a imagem em escala de cinza
img = cv2.imread('Image4.jpeg',0)
rows,cols = img.shape

# cols-1 e rows-1 são os limites das coordenadas. 
#Os dois primeiros parametros são o centro da imagem, ou o centro de rotação.
#O terceiro parâmetro é o angulo de rotação. O ultimo parametro é o fator de
#escala.
M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),25,1)
im_dst = cv2.warpAffine(img,M,(cols,rows))

cv2.imshow('img',im_dst)

