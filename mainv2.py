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

im2 = cv2.imread('Image4.jpeg')
height, width = im2.shape[:2]
im2 = cv2.resize(im2,(int(width/2),int(height/2)),interpolation = cv2.INTER_CUBIC)


#imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
#ret,thresh = cv2.threshold(imgray,127,255,0)
#image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#img = cv2.drawContours(im, contours, -1, (0,255,0), 3)

###COMPRIMIDO 1#########
start_point = (170, 110)
end_point = (240,230)
#color = 255
thickness = 2
# Draw a rectangle with blue line borders of thickness of 2 px
#im = cv2.rectangle(im, start_point, end_point, color, thickness)
comprimido1 = im[110:230,170:240]

vetor_xcm1 = np.arange(len(comprimido1[0]))
vetor_ycm1 = np.arange(len((comprimido1.transpose())[0]))
contador_cm1 = 0
for x in vetor_xcm1:
    for y in vetor_ycm1:
        if comprimido1[y,x] > 180:
            contador_cm1 = contador_cm1+1
print("Comprimido1 =")
print(contador_cm1)
if (contador_cm1 > 2000):
    color = (127,255,0)
    im2 = cv2.rectangle(im2, start_point, end_point, color, thickness)
else:
    color = (255,69,0)
    im2 = cv2.rectangle(im2, start_point, end_point, color, thickness)
    
########################

start_point = (280, 110)
end_point = (350,230)
color = 255
thickness = 2
# Draw a rectangle with blue line borders of thickness of 2 px
im = cv2.rectangle(im, start_point, end_point, color, thickness)
comprimido2 = im[110:230,280:350]

start_point = (380, 110)
end_point = (450,230)
color = 255
thickness = 2
# Draw a rectangle with blue line borders of thickness of 2 px
im = cv2.rectangle(im, start_point, end_point, color, thickness)
comprimido3 = im[110:230,380:450]

start_point = (225, 230)
end_point = (295,340)
color = 255
thickness = 2
# Draw a rectangle with blue line borders of thickness of 2 px
im = cv2.rectangle(im, start_point, end_point, color, thickness)
comprimido4 = im[230:340,225:295]

start_point = (325, 230)
end_point = (395,340)
color = 255
thickness = 2
# Draw a rectangle with blue line borders of thickness of 2 px
im = cv2.rectangle(im, start_point, end_point, color, thickness)
comprimido5 = im[230:340,325:395]

start_point = (225, 360)
end_point = (295,470)
color = 255
thickness = 2
# Draw a rectangle with blue line borders of thickness of 2 px
im = cv2.rectangle(im, start_point, end_point, color, thickness)
comprimido6 = im[360:470,225:295]

start_point = (325, 360)
end_point = (395,470)
color = 255
thickness = 2
# Draw a rectangle with blue line borders of thickness of 2 px
im = cv2.rectangle(im, start_point, end_point, color, thickness)
comprimido7 = im[360:470,325:395]


#######COMPRIMIDO 8 ############
start_point = (170, 470)
end_point = (240,590)
#color = 255
thickness = 2
# Draw a rectangle with blue line borders of thickness of 2 px
#im = cv2.rectangle(im, start_point, end_point, color, thickness)
comprimido8 = im[470:590,170:240]

vetor_xcm1 = np.arange(len(comprimido8[0]))
vetor_ycm1 = np.arange(len((comprimido1.transpose())[0]))
contador_cm1 = 0
for x in vetor_xcm1:
    for y in vetor_ycm1:
        if comprimido8[y,x] > 180:
            contador_cm1 = contador_cm1+1
print("Comprimido8 =")
print(contador_cm1)
if (contador_cm1 > 2000):
    color = (127,255,0)
    im2 = cv2.rectangle(im2, start_point, end_point, color, thickness)
else:
    color = (220,20,60)
    im2 = cv2.rectangle(im2, start_point, end_point, color, thickness)

################################


start_point = (270, 470)
end_point = (340,590)
color = 255
thickness = 2
# Draw a rectangle with blue line borders of thickness of 2 px
im = cv2.rectangle(im, start_point, end_point, color, thickness)
comprimido9 = im[470:590,270:340]

start_point = (370, 470)
end_point = (440,590)
color = 255
thickness = 2
# Draw a rectangle with blue line borders of thickness of 2 px
im = cv2.rectangle(im, start_point, end_point, color, thickness)
comprimido10 = im[470:590,370:440]


vetor_xcm1 = np.arange(len(comprimido1[0]))
vetor_ycm1 = np.arange(len((comprimido1.transpose())[0]))
contador_cm1 = 0
for x in vetor_xcm1:
    for y in vetor_ycm1:
        if comprimido1[y,x] > 180:
            contador_cm1 = contador_cm1+1
print("Comprimido1 =")
print(contador_cm1)

vetor_xcm2 = np.arange(len(comprimido8[0]))
vetor_ycm2 = np.arange(len((comprimido8.transpose())[0]))
contador_cm2 = 0
for x in vetor_xcm2:
    for y in vetor_ycm2:
        if comprimido8[y,x] > 180:
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

 


'''
plt.subplot(2,1,1)
plt.plot(vetor_linha1,linha1,'r')
plt.title('LinhaCentral')



plt.subplot(2,1,2)
plt.plot(vetor_linha1,derivada_linha1,'r')
plt.title('Derivada LinhaCentral')
'''

plt.show()
cv2.imshow("Sample", im)
cv2.imshow("Sample 2", im2)
#cv2.imshow("Area", area_insp)
#cv2.imshow("AreaRotada", area_rotada)
#cv2.imshow("Comprimido1", comprimido1)
#cv2.imshow("Comprimido8", comprimido2)
#cv2.waitKey(0)
#cv2.destroyAllWindows()