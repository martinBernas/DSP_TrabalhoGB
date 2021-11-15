# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 17:40:23 2018

@author: Jean Schmith
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

def detecta_1a_borda(area_insp,corte):
    area_rotada = area_insp.transpose()
    num_linha = int(len(area_rotada[0])/2);
    linha_central = area_insp[num_linha];
    vetor_linha_central = np.arange(len(linha_central))
    pos_ant = linha_central[0]
    retorno = (num_linha,0)

    for i in vetor_linha_central:
        valor = int(linha_central[i]) - int(pos_ant)
        if valor >= corte:
            retorno = (num_linha,i)
            return retorno
        pos_ant = linha_central[i]

def detecta_angulo(area_insp,corte):
    diff = 200
    largura_area_insp = len(area_insp[0])
    altura_area_insp = len(area_insp.transpose()[0])
    area_sup = area_insp[diff:altura_area_insp,0:largura_area_insp]
    area_inf = area_insp[0:(altura_area_insp-diff),0:largura_area_insp]
    ponto_sup = detecta_1a_borda(area_sup,corte)
    ponto_sup_real = (ponto_sup[0],ponto_sup[1])
    
    ponto_inf = detecta_1a_borda(area_inf,corte)
    ponto_inf_real = (ponto_inf[0]+diff,ponto_inf[1])
    
    cat_op = ponto_inf_real[1] - ponto_sup_real[1]
    cat_adj = ponto_inf_real[0]-ponto_sup_real[0]
    angulo = ((np.arctan(cat_op/cat_adj))*180)/np.pi
    angulo - 90
    return angulo
    

im = cv2.imread('Image9.jpeg',0)
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
#im = cv2.rectangle(im, start_point, end_point, color, thickness)
comprimido1 = im[110:230,170:240]

start_point = (170, 470)
end_point = (240,600)
color = (255,0,0)
thickness = 2
# Draw a rectangle with blue line borders of thickness of 2 px
#im = cv2.rectangle(im, start_point, end_point, color, thickness)
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

hist_fullcm1 = cv2.calcHist([comprimido1],[0],None,[256],[0,256])
hist_fullcm2 = cv2.calcHist([comprimido2],[0],None,[256],[0,256])

area_angulo = im[25:775, 25:300]

angulo = detecta_angulo(area_angulo,20)

rows,cols = im.shape
M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),angulo,1)
rotacao_corrigida = cv2.warpAffine(im,M,(cols,rows))


area_borda_lateral = rotacao_corrigida[25:775, 25:300]

borda_lateral = detecta_1a_borda(area_borda_lateral,20)
real_borda_lateral = (borda_lateral[0]+25,borda_lateral[1]+25)
rotacao_corrigida[real_borda_lateral[0],real_borda_lateral[1]] = 255


area_borda_sup = rotacao_corrigida[25:350,25:575].transpose()
borda_sup = detecta_1a_borda(area_borda_sup,20)
real_borda_sup = ( borda_sup[1]+25 , borda_sup[0]+25 )
rotacao_corrigida[real_borda_sup[0],real_borda_sup[1]] = 255

canto_ref = (real_borda_sup[0],real_borda_lateral[1]) # Referencia Sample = 95,155
rotacao_corrigida[canto_ref[0],canto_ref[1]] = 255

ajusteX = canto_ref[0]-95
ajusteY = canto_ref[1]-155 

start_point = (170+ajusteY, 120+ajusteX)
end_point = (240+ajusteY,230+ajusteX)
color = 255
thickness = 2
# Draw a rectangle with blue line borders of thickness of 2 px
rotacao_corrigida = cv2.rectangle(rotacao_corrigida, start_point, end_point, color, thickness)





plt.subplot(2,1,1)
plt.plot(hist_fullcm1)
plt.title('HistogramaCM1')
plt.subplot(2,1,2)
plt.plot(hist_fullcm2)
plt.title('HistogramaCM2')






plt.show()
cv2.imshow("Sample", im)
cv2.imshow("Area_borda_sup", area_borda_sup)
cv2.imshow("Comprimido2", comprimido2)
cv2.imshow("Rotacao corrigida", rotacao_corrigida)

#cv2.waitKey(0)
#cv2.destroyAllWindows()