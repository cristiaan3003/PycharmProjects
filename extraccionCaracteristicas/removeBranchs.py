# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

import numpy as np
from matplotlib import pyplot as plt
from skimage import morphology
import mahotas as mh

# <codecell>
def branchedPoints2(skel):
    X=[]
    #cross X
    X0 = np.array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]])
    X1 = np.array([[1, 0, 1],
                   [0, 1, 0],
                   [1, 0, 1]])
    X.append(X0)
    X.append(X1)
    #T like
    T=[]
    #T0 contains X0
    T0=np.array([[2, 1, 2],
                 [1, 1, 1],
                 [2, 2, 2]])

    T1=np.array([[1, 2, 1],
                 [2, 1, 2],
                 [1, 2, 2]])  # contains X1

    T2=np.array([[2, 1, 2],
                 [1, 1, 2],
                 [2, 1, 2]])

    T3=np.array([[1, 2, 2],
                 [2, 1, 2],
                 [1, 2, 1]])

    T4=np.array([[2, 2, 2],
                 [1, 1, 1],
                 [2, 1, 2]])

    T5=np.array([[2, 2, 1],
                 [2, 1, 2],
                 [1, 2, 1]])

    T6=np.array([[2, 1, 2],
                 [2, 1, 1],
                 [2, 1, 2]])

    T7=np.array([[1, 2, 1],
                 [2, 1, 2],
                 [2, 2, 1]])
    T.append(T0)
    T.append(T1)
    T.append(T2)
    T.append(T3)
    T.append(T4)
    T.append(T5)
    T.append(T6)
    T.append(T7)
    #Y like
    Y=[]
    Y0=np.array([[1, 0, 1],
                 [0, 1, 0],
                 [2, 1, 2]])

    Y1=np.array([[0, 1, 0],
                 [1, 1, 2],
                 [0, 2, 1]])

    Y2=np.array([[1, 0, 2],
                 [0, 1, 1],
                 [1, 0, 2]])

    Y2=np.array([[1, 0, 2],
                 [0, 1, 1],
                 [1, 0, 2]])

    Y3=np.array([[0, 2, 1],
                 [1, 1, 2],
                 [0, 1, 0]])

    Y4=np.array([[2, 1, 2],
                 [0, 1, 0],
                 [1, 0, 1]])
    Y5=np.rot90(Y3)
    Y6 = np.rot90(Y4)
    Y7 = np.rot90(Y5)
    #EE agregado mio
    Y8=np.array([[0, 0, 0],
                 [1, 0, 1],
                 [0, 1, 1]])

    Y.append(Y0)
    Y.append(Y1)
    Y.append(Y2)
    Y.append(Y3)
    Y.append(Y4)
    Y.append(Y5)
    Y.append(Y6)
    Y.append(Y7)
    Y.append(Y8)

    bp = np.zeros(skel.shape, dtype=int)
    for x in X:
        bp = bp + mh.morph.hitmiss(skel,x)
    for y in Y:
        bp = bp + mh.morph.hitmiss(skel,y)
    for t in T:
        bp = bp + mh.morph.hitmiss(skel,t)

    return bp > 0

def branchedPoints(skel):
    branch1=np.array([[2, 1, 2], [1, 1, 1], [2, 2, 2]])
    branch2=np.array([[1, 2, 1], [2, 1, 2], [1, 2, 1]])
    branch3=np.array([[1, 2, 1], [2, 1, 2], [1, 2, 2]])
    branch4=np.array([[2, 1, 2], [1, 1, 2], [2, 1, 2]])
    branch5=np.array([[1, 2, 2], [2, 1, 2], [1, 2, 1]])
    branch6=np.array([[2, 2, 2], [1, 1, 1], [2, 1, 2]])
    branch7=np.array([[2, 2, 1], [2, 1, 2], [1, 2, 1]])
    branch8=np.array([[2, 1, 2], [2, 1, 1], [2, 1, 2]])
    branch9=np.array([[1, 2, 1], [2, 1, 2], [2, 2, 1]])
    br1=mh.morph.hitmiss(skel,branch1)
    br2=mh.morph.hitmiss(skel,branch2)
    br3=mh.morph.hitmiss(skel,branch3)
    br4=mh.morph.hitmiss(skel,branch4)
    br5=mh.morph.hitmiss(skel,branch5)
    br6=mh.morph.hitmiss(skel,branch6)
    br7=mh.morph.hitmiss(skel,branch7)
    br8=mh.morph.hitmiss(skel,branch8)
    br9=mh.morph.hitmiss(skel,branch9)
    return br1+br2+br3+br4+br5+br6+br7+br8+br9

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

def pruning(skeleton, size):
    '''remove iteratively end points "size"
       times from the skeleton
    '''
    for i in range(0, size):
        endpoints = endPoints(skeleton)
        endpoints = np.logical_not(endpoints)
        skeleton = np.logical_and(skeleton,endpoints)
    return skeleton

def cutBranches(bp):
    pointBranch=np.nonzero(bp)
    pointBranch=np.transpose(pointBranch)
    for i in range(0,len(pointBranch)): # marcar los 8 vecinos
        bp[pointBranch[i][0]-1][pointBranch[i][1]-1]=1
        bp[pointBranch[i][0]-1][pointBranch[i][1]]=1
        bp[pointBranch[i][0]-1][pointBranch[i][1]+1]=1
        bp[pointBranch[i][0]][pointBranch[i][1]-1]=1
        bp[pointBranch[i][0]][pointBranch[i][1]+1]=1
        bp[pointBranch[i][0]+1][pointBranch[i][1]-1]=1
        bp[pointBranch[i][0]+1][pointBranch[i][1]]=1
        bp[pointBranch[i][0]+1][pointBranch[i][1]+1]=1
    return bp

def endpoints_top_bottom(l_ep):
    h,w=l_ep.shape
    #bottom
    for j in range(0,h):
        for i in range(0,w):
            if l_ep[j,i]!=0:
                bottom=l_ep[j,i]
                break
     #top
    for j in reversed(range(0,h)):
        for i in range(0,w):
            if l_ep[j,i]!=0:
                top=l_ep[j,i]
                break
    return top,bottom

def remove_branchs(thin, graps=False):
        #a = io.imread("/home/asusn56/PycharmProjects/detect_branched_points/98000080.16.tiff/clase1/clase1_1_98000080.16.tiff")
    #im = a < 255
    #skel = mh.thin(im)
    thinF=thin.copy()
    thin = pruning(thin.copy(),1)
    ep = endPoints(thin)
    if len(np.transpose(np.nonzero(ep)))>2: #si tengo mas de dos end-point hacer el podado
        bp = branchedPoints2(thin)
        edge = thin-bp
        edge_lab,n = mh.label(edge)
        bp_lab,_ = mh.label(bp, np.ones((3,3), bool))#l_bp,_
        ep_lab,_ = mh.label(ep)
        bpCut=cutBranches(bp.copy())
        edgeSeparateBrach=np.logical_not(bpCut)*thin
        edgeSeparateBrach_lab,n_sep = mh.label(edgeSeparateBrach, np.ones((3,3), bool))#label edges
        l_ep=ep*edgeSeparateBrach_lab##Make end-points with the same label than their edge

        ##edges between branched-points
        endpoints_labels = np.where(mh.histogram.fullhistogram(np.uint16(l_ep))[:]==1)[0]
        edges_bp = np.copy(edgeSeparateBrach_lab)
        for l in endpoints_labels:
            edges_bp[np.where(edges_bp == l)]=0
        #edges between end-points and branched points
        edges_ep_to_bp = edgeSeparateBrach_lab * np.logical_not(edges_bp > 0)

        ##edges between branched-points TOP-BOTTOM
        endpoints_labels_top_bottom=endpoints_top_bottom(l_ep)
        edges_bp_top_bottom = np.copy(edgeSeparateBrach_lab)
        for l in endpoints_labels_top_bottom:
            edges_bp_top_bottom[np.where(edges_bp_top_bottom == l)]=0
        #edges between end-points TOP-BOTTOM and branched points
        edges_ep_to_bp_top_bottom = edgeSeparateBrach_lab * np.logical_not(edges_bp_top_bottom > 0)

        #edges between branched points
        edges_bp_to_bp=edgeSeparateBrach_lab*np.logical_not(edges_ep_to_bp.copy())

        thinPruningBranch=np.logical_or(np.logical_or(edges_ep_to_bp_top_bottom,bpCut),edges_bp_to_bp)
        thinF = morphology.skeletonize(thinPruningBranch>0)

        if graps==True:
            plt.figure(1)
            plt.subplot(221,frameon = False, xticks = [], yticks = [])
            plt.title('skeleton')
            plt.imshow(thin,interpolation = 'nearest')
            plt.subplot(223,frameon = False, xticks = [], yticks = [])
            plt.title('skel-bp -> label')
            plt.imshow(edge_lab,interpolation = 'nearest')
            plt.subplot(224,frameon = False, xticks = [], yticks = [])
            plt.title('skel -> end points')
            plt.imshow(ep_lab,interpolation = 'nearest')
            plt.subplot(222,frameon = False, xticks = [], yticks = [])
            plt.title('branched points (bp)')
            plt.imshow(bp_lab,interpolation = 'nearest')

            plt.figure(2)
            plt.imshow(edges_bp_to_bp,interpolation = 'nearest')

            plt.figure(3)
            plt.imshow(bpCut,interpolation = 'nearest')

            plt.figure(4)
            plt.imshow(thinPruningBranch,interpolation = 'nearest')

            plt.figure(5)
            plt.imshow(thinF,interpolation = 'nearest')

            plt.show()
        return thinF




    return thinF
