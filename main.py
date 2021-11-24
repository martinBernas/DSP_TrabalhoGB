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

def desenha_retangulo(imagem, start_point, end_point, color):
    thickness = 2
    retorno = cv2.rectangle(imagem, start_point, end_point, color, thickness)
    return retorno

def verificar_comprimido(comprimido,contagem_minima,limiar_cor):
    vetor_x = np.arange(len(comprimido[0]))
    vetor_y = np.arange(len((comprimido.transpose())[0]))
    contador = 0
    for x in vetor_x:
        for y in vetor_y:
            if comprimido[y,x] > limiar_cor:
                contador = contador+1
            if contador > contagem_minima:
                return 1
            
    return 0
arquivos = ('Image2.jpeg','Image3.jpeg','Image4.jpeg','Image5.jpeg','Image6.jpeg','Image7.jpeg','Image8.jpeg','Image9.jpeg','Image1.jpeg')
for arquivo in arquivos:
    vermelho = (0,0,255);
    verde = (0,255,0);

    im = cv2.imread(arquivo,0)
    im_color = cv2.imread(arquivo)

    height, width = im.shape[:2]
    im = cv2.resize(im,(int(width/2),int(height/2)),interpolation = cv2.INTER_CUBIC)
    im_color = cv2.resize(im_color,(int(width/2),int(height/2)),interpolation = cv2.INTER_CUBIC)

#imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
#ret,thresh = cv2.threshold(imgray,127,255,0)
#image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#img = cv2.drawContours(im, contours, -1, (0,255,0), 3)

####################################################################################################
#Corigindo o angulo da embalagem

    area_angulo = im[25:775, 25:300]

    angulo = detecta_angulo(area_angulo,20)

    rows,cols = im.shape
    M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),angulo,1)
    rotacao_corrigida = cv2.warpAffine(im,M,(cols,rows))
    rotacao_corrigida_color = cv2.warpAffine(im_color,M,(cols,rows))

#####################################################################################################
#Detecta ponto de referencia (canto superior esquerdo) da embalagem
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
#######################################################################################################
#Definir variaveis de ajuste q serão aplicadas sobre as proximas "ferramentas" 
    ajusteX = canto_ref[0]-95
    ajusteY = canto_ref[1]-155 
#######################################################################################################
#Comprimido 1
    start_point = (170+ajusteY, 120+ajusteX)
    end_point = (240+ajusteY,230+ajusteX)

    comprimido1 = rotacao_corrigida[start_point[1]:end_point[1],start_point[0]:end_point[0]]
    presente = verificar_comprimido(comprimido1,2000,180)
    if presente == 1:
        color = verde
    else:
        color = vermelho
    rotacao_corrigida_color = desenha_retangulo(rotacao_corrigida_color,start_point,end_point,color)
#######################################################################################################
#Comprimido 2
    start_point = (280+ajusteY, 110+ajusteX)
    end_point = (350+ajusteY,230+ajusteX)

    comprimido2 = rotacao_corrigida[start_point[1]:end_point[1],start_point[0]:end_point[0]]
    presente = verificar_comprimido(comprimido2,2000,180)
    if presente == 1:
        color = verde
    else:
        color = vermelho
    rotacao_corrigida_color = desenha_retangulo(rotacao_corrigida_color,start_point,end_point,color)
#######################################################################################################
#Comprimido 3
    start_point = (380+ajusteY, 110+ajusteX)
    end_point = (450+ajusteY,230+ajusteX)

    comprimido3 = rotacao_corrigida[start_point[1]:end_point[1],start_point[0]:end_point[0]]
    presente = verificar_comprimido(comprimido3,2000,180)
    if presente == 1:
        color = verde
    else:
        color = vermelho
    rotacao_corrigida_color = desenha_retangulo(rotacao_corrigida_color,start_point,end_point,color)
#######################################################################################################
#Comprimido 4
    start_point = (225+ajusteY, 230+ajusteX)
    end_point = (295+ajusteY,340+ajusteX)

    comprimido4 = rotacao_corrigida[start_point[1]:end_point[1],start_point[0]:end_point[0]]
    presente = verificar_comprimido(comprimido4,2000,180)
    if presente == 1:
        color = verde
    else:
        color = vermelho
    rotacao_corrigida_color = desenha_retangulo(rotacao_corrigida_color,start_point,end_point,color)
#######################################################################################################
#Comprimido 5
    start_point = (325+ajusteY, 230+ajusteX)
    end_point = (395+ajusteY,340+ajusteX)

    comprimido5 = rotacao_corrigida[start_point[1]:end_point[1],start_point[0]:end_point[0]]
    presente = verificar_comprimido(comprimido5,2000,180)
    if presente == 1:
        color = verde
    else:
        color = vermelho
    rotacao_corrigida_color = desenha_retangulo(rotacao_corrigida_color,start_point,end_point,color)
#######################################################################################################
#Comprimido 6
    start_point = (225+ajusteY, 360+ajusteX)
    end_point = (295+ajusteY,470+ajusteX)

    comprimido6 = rotacao_corrigida[start_point[1]:end_point[1],start_point[0]:end_point[0]]
    presente = verificar_comprimido(comprimido6,2000,180)
    if presente == 1:
        color = verde
    else:
        color = vermelho
    rotacao_corrigida_color = desenha_retangulo(rotacao_corrigida_color,start_point,end_point,color)
#######################################################################################################
#Comprimido 7
    start_point = (325+ajusteY, 360+ajusteX)
    end_point = (395+ajusteY,470+ajusteX)
    
    comprimido7 = rotacao_corrigida[start_point[1]:end_point[1],start_point[0]:end_point[0]]
    presente = verificar_comprimido(comprimido7,2000,180)
    if presente == 1:
        color = verde
    else:
        color = vermelho
    rotacao_corrigida_color = desenha_retangulo(rotacao_corrigida_color,start_point,end_point,color)
#######################################################################################################
#Comprimido 8
    start_point = (170+ajusteY, 470+ajusteX)
    end_point = (240+ajusteY,590+ajusteX)

    comprimido8 = rotacao_corrigida[start_point[1]:end_point[1],start_point[0]:end_point[0]]
    presente = verificar_comprimido(comprimido7,2000,180)
    if presente == 1:
        color = verde
    else:
        color = vermelho
    rotacao_corrigida_color = desenha_retangulo(rotacao_corrigida_color,start_point,end_point,color)
#######################################################################################################
#Comprimido 9
    start_point = (270+ajusteY, 470+ajusteX)
    end_point = (340+ajusteY,590+ajusteX)

    comprimido7 = rotacao_corrigida[start_point[1]:end_point[1],start_point[0]:end_point[0]]
    presente = verificar_comprimido(comprimido7,2000,180)
    if presente == 1:
        color = verde
    else:
        color = vermelho
    rotacao_corrigida_color = desenha_retangulo(rotacao_corrigida_color,start_point,end_point,color)
#######################################################################################################
#Comprimido 10
    start_point = (370+ajusteY, 470+ajusteX)
    end_point = (440+ajusteY,590+ajusteX)

    comprimido7 = rotacao_corrigida[start_point[1]:end_point[1],start_point[0]:end_point[0]]
    presente = verificar_comprimido(comprimido7,2000,180)
    if presente == 1:
        color = verde
    else:
        color = vermelho
    rotacao_corrigida_color = desenha_retangulo(rotacao_corrigida_color,start_point,end_point,color)
#######################################################################################################

    cv2.imshow("Imagem_inicial", im)
    cv2.imshow("Rotacao corrigida", rotacao_corrigida)
    cv2.imshow("Resultado", rotacao_corrigida_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#cv2.waitKey(0)
#cv2.destroyAllWindows()