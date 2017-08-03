import order_points as op
import numpy as np
import math
import cv2
import removeBranchs as rb
import mahotas as mh
import matplotlib.pyplot as plt
from scipy import signal


def vector_perpendicular(Ax,Ay,Bx,By,length):
        v_x=Bx-Ax
        v_y=By-Ay
        mag = math.sqrt(v_x*v_x + v_y*v_y)
        v_x = v_x / mag
        v_y = v_y / mag
        temp = v_x
        v_x = -v_y
        v_y = temp
        #C.x = B.x + v.x * length; C.y = B.y + v.y * length;
        #length=20
        C_x_1 = int(round(Ax +v_x * length))
        C_y_1 = int(round(Ay + v_y * length))
        #length=-20
        C_x_2 =int(round( Ax + v_x *(-1)*length))
        C_y_2 =int(round( Ay + v_y *(-1)*length))
        return C_x_1,C_y_1,C_x_2,C_y_2

def lineas_perpendiculares(thin,thresh,vent,length):#thin img [0,255]
    #thin=cv2.copyMakeBorder(thin, top=15, bottom=15, left=15, right=15, borderType= cv2.BORDER_CONSTANT, value=0 )
    #thresh=cv2.copyMakeBorder(thresh, top=15, bottom=15, left=15, right=15, borderType= cv2.BORDER_CONSTANT, value=255 )
    aux_smooth_thin=thin.copy()
    aux_smooth_thin[aux_smooth_thin==255]=1 #img [0,1]
    sk_puntos=op.order_points(aux_smooth_thin.copy(),True)

    #eliminar algunos puntos repetidos que me retorna order_points
    sk_puntos = [list(row) for row in sk_puntos]#http://stackoverflow.com/questions/16296643/convert-tuple-to-list-and-back
    import itertools
    sk_puntos=list(sk_puntos for sk_puntos,_ in itertools.groupby(sk_puntos))#http://stackoverflow.com/questions/2213923/python-removing-duplicates-from-a-list-of-lists


    if len(sk_puntos)<vent:
        ventana=int(round(len(sk_puntos)/2))
        ventana_inv=len(sk_puntos)-ventana
    else:
        dd=1/vent
        ventana=int(round(len(sk_puntos)*dd))
        ventana_inv=len(sk_puntos)-ventana

    list_coor_ep_line=[]
    #print(sk_puntos)
    #print(ventana)
    #print(ventana_inv)
    #print(vent)
    #print(len(sk_puntos))

    for i in range(0,len(sk_puntos)-ventana,1):
        aux=np.zeros_like(thin)
        #v.x = B.x - A.x; v.y = B.y - A.y;
        Ax=sk_puntos[i][1]
        Ay=sk_puntos[i][0]
        Bx=sk_puntos[i+ventana][1]
        By=sk_puntos[i+ventana][0]
        C_x_1,C_y_1,C_x_2,C_y_2=vector_perpendicular(Ax,Ay,Bx,By,length)
        #Mask line
        #para cortar la linea al largo transversal del cromosoma
        cv2.line(aux,(C_x_1,C_y_1),(C_x_2,C_y_2),255,1)
        enmascarada=cv2.bitwise_and(aux.copy(),cv2.bitwise_not(thresh.copy()))#ajusto linea al ancho cromosoma
        enmascarada[enmascarada==255]=1
        ep=rb.endPoints(enmascarada)#obtengo end-ponts perpendicular line
        coor_ep_line=np.transpose(np.nonzero(ep))
        list_coor_ep_line=np.append(list_coor_ep_line,coor_ep_line)

    inv_sk_puntos=list(reversed(sk_puntos))
    list_coor_ep_line2=[]
    for i in range(0,len(inv_sk_puntos)-ventana_inv):
        aux=np.zeros_like(thin)
        #v.x = B.x - A.x; v.y = B.y - A.y;
        Ax=inv_sk_puntos[i][1]
        Ay=inv_sk_puntos[i][0]
        Bx=inv_sk_puntos[i-ventana_inv][1]
        By=inv_sk_puntos[i-ventana_inv][0]
        C_x_1,C_y_1,C_x_2,C_y_2=vector_perpendicular(Ax,Ay,Bx,By,length)
        #Mask line
        #para cortar la linea al largo transversal del cromosoma
        cv2.line(aux,(C_x_1,C_y_1),(C_x_2,C_y_2),255,1)
        enmascarada=cv2.bitwise_and(aux.copy(),cv2.bitwise_not(thresh.copy()))#ajusto linea al ancho cromosoma
        enmascarada[enmascarada==255]=1
        ep=rb.endPoints(enmascarada)#obtengo end-ponts perpendicular line
        coor_ep_line=np.transpose(np.nonzero(ep))
        list_coor_ep_line2=np.append(list_coor_ep_line2,coor_ep_line)


    list_coor_ep_line=list(zip(*[iter(list_coor_ep_line)]*4))
    list_coor_ep_line2=list(zip(*[iter(list_coor_ep_line2)]*4))

    list_coor_ep_line=list_coor_ep_line+list(reversed(list_coor_ep_line2))

    #aux=np.zeros_like(thin)
    #aux=thin.copy()
    #lista=list(list_coor_ep_line)
    #for i in range(0,len(lista)):
    #    cv2.line(aux,(int(lista[i][1]),int(lista[i][0])),(int(lista[i][3]),int(lista[i][2])),255,1)
    #    cv2.namedWindow("img1",cv2.WINDOW_NORMAL)
    #    cv2.imshow("img1",aux)
    #    cv2.waitKey(0)

    return list_coor_ep_line

def dProfile(img,thin,l_perp):
    #Median Blurring
    median = cv2.medianBlur(img.copy(),5)
    D = []
    length_lineas=[]#largo en pixeles de cada una de las lineas perpendiculares
    min=1000000

    #cv2.namedWindow("img",cv2.WINDOW_NORMAL)
    #cv2.imshow("img",img)
    #cv2.namedWindow("imgBLU",cv2.WINDOW_NORMAL)
    #cv2.imshow("imgBLU",median)
    #cv2.namedWindow("imgTHIN",cv2.WINDOW_NORMAL)
    #cv2.imshow("imgTHIN",thin)
    #cv2.namedWindow("median-THIN",cv2.WINDOW_NORMAL)
    #cv2.imshow("median-THIN",cv2.bitwise_and(median,cv2.bitwise_not(thin)))
    #cv2.waitKey(0)


    for i in range(0,len(l_perp)):
        mask_linea=np.zeros_like(img)
        cv2.line(mask_linea,(int(l_perp[i][1]),int(l_perp[i][0])),(int(l_perp[i][3]),int(l_perp[i][2])),255,1)
        mask_linea=cv2.bitwise_and(mask_linea,cv2.bitwise_not(thin))
        n=len(np.transpose(np.nonzero(mask_linea)))
        #print("n: "+str(n))
        if n<min and i>(len(l_perp)*0.1) and i<(len(l_perp)*0.9):
            centromero_coord=l_perp[i]
            min=n
        linea=cv2.bitwise_and(median,mask_linea)
        suma=np.sum(linea)
        #print("suma: "+str(suma))
        g=int(round(suma/n))
        #print(g)
        D=np.append(D,g)


    #    cv2.namedWindow("imgTHIN",cv2.WINDOW_NORMAL)
    #    cv2.imshow("imgTHIN",mask_linea)
    #    cv2.waitKey(0)
    # for i in range(0,len(lineas_perpendiculares)):
    #     cv2.line(mask_linea,(puntos_perpendiculares[i][0],puntos_perpendiculares[i][1]),(puntos_perpendiculares[i][2],puntos_perpendiculares[i][3]),255,1)
    #     masked_linea=cv2.bitwise_and(cv2.bitwise_not(thresh),mask_linea)
    #     gray_linea=cv2.bitwise_and(gray,masked_linea)
    #     n=len(np.transpose(np.nonzero(gray_linea))) #cantidad de puntos de la linea
    #     D_i=cv2.mean(gray_linea,mask_linea)[0] #media o promedio ?
    #     D.append([D_i,n])
    #     #clear variables
    #     mask_linea=np.zeros(gray.shape,np.uint8)
    #     gray_linea=mask_linea
    #     masked_linea=mask_linea
    return D,centromero_coord



def dProfile_and_centromere_pos(img,thin,thresh):
    ventanaDP=10
    lengthDP=20#largo de la lineas perpendiculares al thin
    lineas_per=lineas_perpendiculares(thin.copy(),thresh.copy(),ventanaDP,lengthDP)
    D,centromere_coord=dProfile(img,thin,lineas_per) #density profile

    return D,centromere_coord

def re_sampled(dP,N):
    M=len(dP)
    re_dP=[]
    for i in range(0,N):
        j=math.trunc(i*((M-1)/(N-1))+((N-M)/(N-1)))
        #print(j)
        re_dP.append(dP[j])
    return re_dP

def gProfile(dP):
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    dP=np.uint8(dP)
    # Gradient-Y
    grad_y = cv2.Sobel(dP,ddepth,0,1,ksize = 5, scale = scale, delta = delta, borderType = cv2.BORDER_DEFAULT)
    #plt.plot(grad_y)
    #plt.show()
    return grad_y

def numBands_p_arm_q_arm(signP,centromere_pos):
   pos=int(round((centromere_pos[0]+centromere_pos[2])/2))
   signP_p_arm=signP[:pos]
   signP_q_arm=signP[pos:]
   numBands_p_arm=((signP_p_arm[:-1] * signP_p_arm[1:]) < 0).sum()
   numBands_q_arm=((signP_q_arm[:-1] * signP_q_arm[1:]) < 0).sum()
   return numBands_p_arm,numBands_q_arm

def numBandsBlackWhite(signP):
    signP=np.transpose(signP)
    signP=signP[0][:]
    countBlack=0
    countWhite=0
    for i in range(0,len(signP)-1):
        if signP[i]>0 and signP[i+1]<0:
            countBlack=countBlack+1
        if signP[i]<0 and signP[i+1]>0:
            countWhite=countWhite+1
    return countBlack,countWhite

def numBandsBlack_q_arm(signP,centromere_pos):
   pos=int(round((centromere_pos[0]+centromere_pos[2])/2))
   signP=np.transpose(signP)
   signP=signP[0][:]
   signP_q_arm=signP[pos:]
   countBlack=0
   countWhite=0
   for i in range(0,len(signP_q_arm)-1):
        if signP_q_arm[i]>0 and signP_q_arm[i+1]<0:
            countBlack=countBlack+1
        if signP_q_arm[i]<0 and signP_q_arm[i+1]>0:
            countWhite=countWhite+1
   return countBlack,countWhite

