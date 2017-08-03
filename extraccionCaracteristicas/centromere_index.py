import cv2
import numpy as np
import mahotas as mh
from matplotlib import pyplot as plt



def centromere_index_ratio_of_the_length(smooth_thin_extend,centromere_pos):
    line=np.zeros_like(smooth_thin_extend)
    line=cv2.line(line,(int(centromere_pos[1]),int(centromere_pos[0])),(int(centromere_pos[3]),int(centromere_pos[2])),255,1)
    mask=cv2.bitwise_and(cv2.bitwise_xor(smooth_thin_extend,line),smooth_thin_extend)
    labeling,n=mh.label(mask,np.ones((3,3)))
    #plt.figure(1)
    #plt.title('skel-bp -> label')
    #plt.imshow(labeling,interpolation = 'nearest')
    aux_labeling=labeling.copy()
    aux_labeling[aux_labeling==1]=255
    aux_labeling[aux_labeling==2]=0
    #CI(L)= L_p /(L_p + L_q )
    L_p=len(np.transpose(np.nonzero(aux_labeling)))
    aux_labeling=labeling.copy()
    aux_labeling[aux_labeling==1]=0
    aux_labeling[aux_labeling==2]=255
    L_q=len(np.transpose(np.nonzero(aux_labeling)))
    CI_L=L_p/(L_p+L_q)
    return CI_L


def centromere_index_ratio_of_the_area(thresh,centromere_pos):
    line=np.zeros_like(thresh)
    line=cv2.line(line,(int(centromere_pos[1]),int(centromere_pos[0])),(int(centromere_pos[3]),int(centromere_pos[2])),255,3)
    mask=cv2.bitwise_not(cv2.bitwise_or(thresh,line))
    labeling,n=mh.label(mask,np.ones((3,3)))
    #plt.figure(1)
    #plt.title('skel-bp -> label')
    #plt.imshow(labeling,interpolation = 'nearest')
    #plt.show()
    #cv2.namedWindow("img",cv2.WINDOW_NORMAL)
    #cv2.imshow("img",mask)
    #print(n)
    #cv2.waitKey(0)
    aux_labeling=labeling.copy()
    aux_labeling[aux_labeling==1]=255
    aux_labeling[aux_labeling==2]=0
    #CI(A)= A_p /(A_p + A_q )
    A_p=len(np.transpose(np.nonzero(aux_labeling)))
    aux_labeling=labeling.copy()
    aux_labeling[aux_labeling==1]=0
    aux_labeling[aux_labeling==2]=255
    A_q=len(np.transpose(np.nonzero(aux_labeling)))
    CI_A=A_p/(A_p+A_q)
    return CI_A

