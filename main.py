# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 17:40:23 2018

@author: Jean Schmith
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

im = cv2.imread('Image4.jpeg',0)
height, width = im.shape[:2]
im = cv2.resize(im,(int(width/2),int(height/2)),interpolation = cv2.INTER_CUBIC)


#imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
#ret,thresh = cv2.threshold(imgray,127,255,0)
#image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#img = cv2.drawContours(im, contours, -1, (0,255,0), 3)

start_point = (170, 110)
end_point = (240,230)
color = 255
thickness = 2
# Draw a rectangle with blue line borders of thickness of 2 px
im = cv2.rectangle(im, start_point, end_point, color, thickness)
comprimido1 = im[110:230,170:240]

start_point = (170, 470)
end_point = (240,600)
color = 255
thickness = 2
# Draw a rectangle with blue line borders of thickness of 2 px
im = cv2.rectangle(im, start_point, end_point, color, thickness)
comprimido2 = im[470:600,170:240]
vetor_xcm1 = np.arange(len(comprimido1[0]))
vetor_ycm1 = np.arange(len((comprimido1.transpose())[0]))
contador_cm1 = 0
for x in vetor_xcm1:
    for y in vetor_ycm1:
        if comprimido1[y,x] > 180:
            contador_cm1 = contador_cm1+1
print("Comprimido1 =")
print(contador_cm1)

vetor_xcm2 = np.arange(len(comprimido2[0]))
vetor_ycm2 = np.arange(len((comprimido2.transpose())[0]))
contador_cm2 = 0
for x in vetor_xcm2:
    for y in vetor_ycm2:
        if comprimido2[y,x] > 180:
            contador_cm2 = contador_cm2+1
print("Comprimido2 =")
print(contador_cm2)






area_insp = im[25:300, 25:575]

linha1 = area_insp.transpose()[287]
vetor_linha1 = np.arange(len(linha1))
derivada_linha1 = np.zeros(len(linha1),int)

pos_ant = linha1[0]

for i in vetor_linha1:
    valor = int(linha1[i]) - int(pos_ant)
    derivada_linha1[i] = valor
    pos_ant = linha1[i]
    
area_rotada = area_insp.transpose()

     
    

 



plt.subplot(2,1,1)
plt.plot(vetor_linha1,linha1,'r')
plt.title('LinhaCentral')



plt.subplot(2,1,2)
plt.plot(vetor_linha1,derivada_linha1,'r')
plt.title('Derivada LinhaCentral')


plt.show()
cv2.imshow("Sample", im)
cv2.imshow("Area", area_insp)
cv2.imshow("AreaRotada", area_rotada)
cv2.imshow("Comprimido1", comprimido1)
cv2.imshow("Comprimido2", comprimido2)
#cv2.waitKey(0)
#cv2.destroyAllWindows()