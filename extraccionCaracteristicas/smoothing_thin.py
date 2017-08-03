
import order_points as op
from matplotlib import pyplot as plt
import cv2
import numpy as np

#thin-> debe ser una imagen binaria normalizada a [0,1]
#step-> paso. a mayor paso mas smooth
def smoothing_thin(thin,step):
    #retorna la lista de puntos ordenados desde el primer entre el primer y ultimo end-point
    list_order_points=op.order_points(thin)
    #print(list_order_points)
    #smooth step 5 points-> list of position index
    smoothPointsIndexs=list(np.arange(0,len(list_order_points),step))
    #add end-point index
    smoothPointsIndexs=np.insert(smoothPointsIndexs,len(smoothPointsIndexs),len(list_order_points)-1)
    #check if end-point index is not repeat
    #if repeat delete one
    #if (smoothPointsIndexs[len(smoothPointsIndexs)-1]==smoothPointsIndexs[len(smoothPointsIndexs)-2]):
    #    smoothPointsIndexs=smoothPointsIndexs[0:len(smoothPointsIndexs)-1]

    #list of coordinate point
    smooth_list_order_points=np.zeros_like(list_order_points)
    smooth_list_order_points=smooth_list_order_points[0:len(smoothPointsIndexs)]
    for i in range(0,len(smoothPointsIndexs)):
        smooth_list_order_points[i][0]=list_order_points[smoothPointsIndexs[i]][0]
        smooth_list_order_points[i][1]=list_order_points[smoothPointsIndexs[i]][1]

    #imagen de los puntos extraidos sin conectarlos
    smooth_thin_points=np.zeros_like(thin)
    for i in range(0,len(smooth_list_order_points)):
        smooth_thin_points[smooth_list_order_points[i][0]][smooth_list_order_points[i][1]]=255

    #imagen de los puntos extraidos conectados mediantes lineas rectas
    smooth_thin=np.zeros_like(thin)
    for i in range(0,len(smooth_list_order_points)-1):
        #print(smooth_list_order_points[i])
        cv2.line(smooth_thin,(smooth_list_order_points[i][1],smooth_list_order_points[i][0]),(smooth_list_order_points[i+1][1],smooth_list_order_points[i+1][0]),255,1)

    return smooth_thin



