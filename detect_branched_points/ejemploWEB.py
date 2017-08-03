# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import cv2
from skimage import io
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
    Y.append(Y0)
    Y.append(Y1)
    Y.append(Y2)
    Y.append(Y3)
    Y.append(Y4)
    Y.append(Y5)
    Y.append(Y6)
    Y.append(Y7)

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
    print(l_ep)
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
#====================================================
#=====================================================
#=======================================================

# <codecell
if __name__ == "__main__":
    #a = io.imread("/home/asusn56/PycharmProjects/detect_branched_points/skeletos/img2.tiff")
    thin = io.imread("/home/asusn56/PycharmProjects/detect_branched_points/skeletos/img19.tiff")
    print(thin)
    pruned_thin=remove_branchs(thin,True)












# <codecell>



# <codecell>
'''endpoints = np.zeros(skel.shape)
edges = np.zeros(skel.shape)

for l in range(1,n+1):
    cur_edge = edge_lab == l
    cur_end_points = endPoints(cur_edge)
    pruned_edge = pruning(cur_edge,1)
    edges = np.logical_or(pruned_edge, edges)
    endpoints = np.logical_or(endpoints,cur_end_points)

lab_bp, nbp = mh.label(bp)
lab_ep, nep = mh.label(endpoints+ep)
pruned_edges, ned = mh.label(edges)

# <codecell>

plt.subplot(321,frameon = False, xticks = [], yticks = [])
plt.title(str(np.unique(skel)))
plt.imshow(skel, interpolation='nearest')

plt.subplot(322,frameon = False, xticks = [], yticks = [])
plt.title(str(np.unique(lab_bp)))
plt.imshow(lab_bp, interpolation='nearest')

plt.subplot(323,frameon = False, xticks = [], yticks = [])
plt.title(str(np.unique(edge_lab)))
plt.imshow(edge_lab, interpolation='nearest')

plt.subplot(326,frameon = False, xticks = [], yticks = [])
edge_mask = lab_ep>0
ep_edgelab = edge_lab*edge_mask
plt.title(str(np.unique(ep_edgelab)))
plt.imshow(ep_edgelab,interpolation='nearest')#lab_ep

plt.subplot(324,frameon = False, xticks = [], yticks = [])
plt.title(str(np.unique(pruned_edges)))
plt.imshow(pruned_edges, interpolation='nearest')

plt.subplot(325,frameon = False, xticks = [], yticks = [])
plt.title(str(np.unique(lab_ep)))
plt.imshow(lab_ep, interpolation='nearest')


plt.show()
# <headingcell level=3>

# First make vertex from branched points and then link to the neighbouring endpoints

# <codecell>

BPlabel=np.unique(lab_bp)[1:]
EPlabel=np.unique(lab_ep)[1:]
EDlabel=np.unique(pruned_edges)[1:]

G=nx.Graph()
node_index=BPlabel.max()
selected_ep_labels = []
for bpl in BPlabel:
    #branched point location
    bp_row,bp_col= np.where(lab_bp==bpl)[0][0], np.where(lab_bp==bpl)[1][0]
    bp_neighbor= lab_ep[bp_row-1:bp_row+2,bp_col-1:bp_col+2]

    G.add_node(bpl,kind='BP',row=bp_row,col=bp_col,label=bpl)
    #branched point neighborhood
    neig_in_EP=lab_ep[bp_row-1:bp_row+2,bp_col-1:bp_col+2]
    neig_inEplabel=np.unique(neig_in_EP)[1:]
    print ('bp label',bpl,' pos',(bp_row,bp_col), 'ep neighbor label:',neig_inEplabel)

    for epl in neig_inEplabel:
        selected_ep_labels.append(epl)
        node_index = node_index+1
        #print 'bp label',bpl,' pos',(bp_row,bp_col),neig_inEplabel, 'current ep',epl
        ep_row,ep_col= np.where(lab_ep==epl)[0][0], np.where(lab_ep==epl)[1][0]
        G.add_node(node_index,kind='EP',row=ep_row,col=ep_col,label=epl)
        G.add_edge(bpl,node_index, weight=1)
        print( 'bp label',bpl,':',(bp_row,bp_col),neig_inEplabel,' -ep',epl,'(',node_index,')',':',(ep_row,ep_col),)
        #print 'edge',(bpl, node_index)

# <codecell>

plt.figsize(5,7)
nx.draw(G)

# <codecell>

#Add isolated endpoints in the Graph
#label of isolated end points
lonely_ep = list(set(EPlabel) - set(selected_ep_labels))
#lep:lonely ep
for lep in lonely_ep:
    lep_row,lep_col= np.where(lab_ep==lep)[0][0], np.where(lab_ep==lep)[1][0]
    node_index = node_index+1
    G.add_node(node_index,kind='EP',row=lep_row,col=lep_col,label=lep)

#print G.node[1]
#edges dict
edges={}
for ed in EDlabel:
    edges[ed] = np.where(pruned_edges==ed),np.sum(pruned_edges==ed)
    #print edges[ed]
#get nodes of kind EP

ep_nodes=[node for node,data in G.nodes(data=True) if data['kind'] == 'EP']
print (ep_nodes)
#print edges.keys()
ep_edges_lab = (lab_ep>0)*edge_lab
#print G.nodes()
#

# <codecell>

plt.figsize(5,8)
plt.title('Isolated endpoints are added')
nx.draw(G)

# <headingcell level=3>

# each endpoints pairs are linked

# <codecell>

# get all nodes of kind EP
#get their label
node_to_label={}
label_to_node={}
for node,data in G.nodes(data=True):
    if data['kind']=='EP':
        #print node,data['label']
        label_to_node[data['label']]=node
print (label_to_node)

# <codecell>

#ep_labels from edge lab -> ep labels directly!
#
# node_index is not endpoint label!
#
map_epEdlab_epPair={}
print (EDlabel)
##plt.figsize(14,8)
for elab in EDlabel:
    #subplot(3,3,elab)
    mask = (ep_edges_lab==elab)# ep lab=edges lab
    nodes_pair = np.unique(mask*lab_ep)[1:]
    pair_weight = np.sum(pruned_edges==elab)
    edge_image = (pruned_edges==elab)*elab
    print ('edge lab',elab, 'ep2 lab',nodes_pair,'weight',pair_weight,'node1:',nodes_pair[0])
    ##title(str(elab)+str(np.where(ep_edgelab==elab))+str(nodes_pair))
    ##imshow(mask, interpolation='nearest')
    node1 = label_to_node[nodes_pair[0]]
    node2 = label_to_node[nodes_pair[1]]
    G.add_edge(node1, node2, weight = pair_weight, image=edge_image)

# <codecell>

plt.figsize(5,8)
pos=nx.spring_layout(G)
#nx.draw_circular(G)
#subplot(121)
#nx.draw(G, width=range(1,4))
#imshow(edge_lab,interpolation='nearest')

#subplot(122)
edge_labels=dict([((u,v,),d['weight']) for u,v,d in G.edges(data=True)])
nx.draw(G)
#nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)

'''
