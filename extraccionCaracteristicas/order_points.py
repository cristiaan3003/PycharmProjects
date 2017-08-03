import numpy as np
from scipy import ndimage as nd
#import skimage as
import mahotas as mh

def endPoints(skel):
    endpoint1=np.array([[0, 0, 0],
                        [0, 1, 0],
                        [2, 1, 2]])

    endpoint2=np.array([[0, 0, 0],
                        [0, 1, 2],
                        [0, 2, 1]])

    endpoint3=np.array([[0, 0, 2],
                        [0, 1, 1],
                        [0, 0, 2]])

    endpoint4=np.array([[0, 2, 1],
                        [0, 1, 2],
                        [0, 0, 0]])

    endpoint5=np.array([[2, 1, 2],
                        [0, 1, 0],
                        [0, 0, 0]])

    endpoint6=np.array([[1, 2, 0],
                        [2, 1, 0],
                        [0, 0, 0]])

    endpoint7=np.array([[2, 0, 0],
                        [1, 1, 0],
                        [2, 0, 0]])

    endpoint8=np.array([[0, 0, 0],
                        [2, 1, 0],
                        [1, 2, 0]])

    ep1=mh.morph.hitmiss(skel,endpoint1)
    ep2=mh.morph.hitmiss(skel,endpoint2)
    ep3=mh.morph.hitmiss(skel,endpoint3)
    ep4=mh.morph.hitmiss(skel,endpoint4)
    ep5=mh.morph.hitmiss(skel,endpoint5)
    ep6=mh.morph.hitmiss(skel,endpoint6)
    ep7=mh.morph.hitmiss(skel,endpoint7)
    ep8=mh.morph.hitmiss(skel,endpoint8)
    ep = ep1+ep2+ep3+ep4+ep5+ep6+ep7+ep8
    return ep


def order_points(im, C8=False):
    #total number of pixels in the curve
    pixel=np.array([-1,-1])
    pixN=np.sum(im==1)
    pixelsList=[]
    #get end point by H&Miss op by ndimage
    #print scipyEndPoints(im)
    #get end points by hit&miss operator provided by mahotas
    #ep=bep.get_endpoints(im)
    #ep=scipyEndPoints(im)
    ep=endPoints(im)
    lab,n=nd.label(ep)
    #where the first endpoint is
    first_indices=np.where(lab==1)
    pixel[0]=first_indices[0][0]
    pixel[1]=first_indices[1][0]
    pixelsList.append(np.copy(pixel))
    #print "first point",first_indices," vector",walkingPixel
    firstEndPoint=np.uint8(lab==1)
    #keep an image of the second end point
    lastEndPoint=np.uint8(lab==2)
    last_Indices=np.where(lastEndPoint==1)
    #print "last point",last_Indices
    ######################################################
    ## walk on curve with a 3x3 neighborhood
    ######################################################
    #Init
    current_curve=np.copy(im)
    current_point=np.copy(firstEndPoint)
    ###start to walk on curve
    #remove the second last point
    #current_curve=current_curve-lastEndPoint##too heavy?, try indices
    current_curve[last_Indices[0][0],last_Indices[1][0]]=0
    c_point_ind=np.where(current_point==1)
    #print c_point_ind
    li=c_point_ind[0][0]
    col=c_point_ind[1][0]
    for i in range (0,pixN-2):
        #3x3 neighborhood arround the endpoint
        neighbor=current_curve[li-1:li+2,col-1:col+2]
        neighbor[1,1]=0
        #control-> SIEMPRE TIENE QUE TENER AL MENOS UN VECINO
        if np.sum(neighbor==1)==0:
            neighbor[1,1]=1
        #################
        #Can only handle a curve
        #such np.where(neihgbor==1) must gives only one pixel
        #############
        nextPointIndices=np.where(neighbor==1)##vectO'M=nextPointIndices
        ##remove the first point from the curve
        current_curve[li,col]=0
        pixN=pixN-1
        #print current_curve
        ##compute nextPoint indices in the original base vectOM=OO'+O'M
        li=(li-1)+nextPointIndices[0][0]
        col=(col-1)+nextPointIndices[1][0]
        pixel[0]=li
        pixel[1]=col
        pixelsList.append(np.copy(pixel))
    #don't forget the last pixel
    pixel[0]=last_Indices[0][0]
    pixel[1]=last_Indices[1][0]
    pixelsList.append(np.copy(pixel))

    return pixelsList
