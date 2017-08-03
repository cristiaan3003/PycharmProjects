import numpy as np
from skimage import img_as_ubyte
import cv2
import extras
import math
import order_points as op
import smoothing_thin as smooth
import removeBranchs as rb

def neighbours(x,y,image):
    "Return 8-neighbours of image point P1(x,y), in a clockwise order"
    img = image
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
    return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],     # P2,P3,P4,P5
                img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]    # P6,P7,P8,P9

def transitions(neighbours):
    "No. of 0,1 patterns (transitions from 0 to 1) in the ordered sequence"
    n = neighbours + neighbours[0:1]      # P2, P3, ... , P8, P9, P2
    return sum( (n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]) )  # (P2,P3), (P3,P4), ... , (P8,P9), (P9,P2)

def zhangSuen(image):
    "the Zhang-Suen Thinning Algorithm"
    Image_Thinned = image.copy()  # deepcopy to protect the original image
    changing1 = changing2 = 1        #  the points to be removed (set as 0)
    while changing1 or changing2:   #  iterates until no further changes occur in the image
        # Step 1
        changing1 = []
        rows, columns = Image_Thinned.shape               # x for rows, y for columns
        for x in range(1, rows - 1):                     # No. of  rows
            for y in range(1, columns - 1):            # No. of columns
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1     and    # Condition 0: Point P1 in the object regions
                    2 <= sum(n) <= 6   and    # Condition 1: 2<= N(P1) <= 6
                    transitions(n) == 1 and    # Condition 2: S(P1)=1
                    P2 * P4 * P6 == 0  and    # Condition 3
                    P4 * P6 * P8 == 0):         # Condition 4
                    changing1.append((x,y))
        for x, y in changing1:
            Image_Thinned[x][y] = 0
        # Step 2
        changing2 = []
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1   and        # Condition 0
                    2 <= sum(n) <= 6  and       # Condition 1
                    transitions(n) == 1 and      # Condition 2
                    P2 * P4 * P8 == 0 and       # Condition 3
                    P2 * P6 * P8 == 0):            # Condition 4
                    changing2.append((x,y))
        for x, y in changing2:
            Image_Thinned[x][y] = 0
    return Image_Thinned

def extend_thin(thin,thresh,ventana):
    #obtener contorno del humbral para testear si el
    #punto a agregar esta dentro o fuera -> si esta afuera se debe cortar
    _, contours, _ = cv2.findContours(cv2.bitwise_not(thresh),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #retorna la lista de puntos ordenados desde el primer entre el primer al ultimo end-point
    aux_smooth_thin=thin.copy()
    aux_smooth_thin[aux_smooth_thin==1]=255
    #cv2.namedWindow("im",cv2.WINDOW_NORMAL)
    #cv2.imshow("im",aux_smooth_thin)

    sk_puntos=list(op.order_points(thin.copy()))

     #eliminar algunos puntos repetidos que me retorna order_points
    sk_puntos = [list(row) for row in sk_puntos]#http://stackoverflow.com/questions/16296643/convert-tuple-to-list-and-back
    import itertools
    sk_puntos=list(sk_puntos for sk_puntos,_ in itertools.groupby(sk_puntos))#http://stackoverflow.com/questions/2213923/python-removing-duplicates-from-a-list-of-lists

    nn=int(round(len(sk_puntos)/ventana))
    if nn==0 or nn==1:
        nn=2

    count=0
    while True:#extender puntos superiores
        Ax=sk_puntos[0][1]
        Ay=sk_puntos[0][0]
        Bx=sk_puntos[nn][1]
        By=sk_puntos[nn][0]
        v_x=Bx-Ax
        v_y=By-Ay
        mag = math.sqrt(v_x*v_x + v_y*v_y)
        v_x = v_x / mag
        v_y = v_y / mag
        length=-1
        C_x_1 = int(round(Ax + v_x * length))
        C_y_1 = int(round(Ay + v_y * length))
        if count==0:
            auxCx=C_x_1
            auxCy=C_y_1
        #evaluar si el punto esta dentro del contorno
        #si esta dentro del contorno lo agrego y sigo para agregar uno mas
        #si no esta dentro del contorno corto y salgo del bucle
        #False, it finds whether the point is inside or outside or on the contour (it returns +1, -1, 0 respectively).
        dist = cv2.pointPolygonTest(contours[0],(C_x_1,C_y_1),False)
        if dist==-1 or dist==0: #cortar while
            break
        elif (dist==1 and auxCx==C_x_1 and auxCy==C_y_1 and count!=0):
            break
        count=count+1
        sk_puntos.insert(0,[C_y_1,C_x_1])

    count=0



    while True:#extender puntos inferiores
        Ax=sk_puntos[len(sk_puntos)-1][1]
        Ay=sk_puntos[len(sk_puntos)-1][0]
        Bx=sk_puntos[len(sk_puntos)-nn][1]
        By=sk_puntos[len(sk_puntos)-nn][0]
        v_x=Bx-Ax
        v_y=By-Ay
        mag = math.sqrt(v_x*v_x + v_y*v_y)
        #print(mag)
        v_x = v_x / mag
        v_y = v_y / mag
        #print(v_x)
        #print(v_y)
        length=-1
        C_x_1 = int(round(Ax + v_x * length))
        C_y_1 = int(round(Ay + v_y * length))
        #print(C_x_1)
        #print(C_y_1)
        #print(count)
        if count==0:
            auxCx=C_x_1
            auxCy=C_y_1
        #evaluar si el punto esta dentro del contorno
        #si esta dentro del contorno lo agrego y sigo para agregar uno mas
        #si no esta dentro del contorno corto y salgo del bucle
        #False, it finds whether the point is inside or outside or on the contour (it returns +1, -1, 0 respectively).
        dist = cv2.pointPolygonTest(contours[0],(C_x_1,C_y_1),False)
        #print(dist)
        if dist==-1 or dist==0: #cortar while-> si esta afuera o sobre
            break
        elif (dist==1 and auxCx==C_x_1 and auxCy==C_y_1 and count!=0): #cortar si el punto anterior es igual al actual
            break
        count=count+1
        sk_puntos.insert(len(sk_puntos),[C_y_1,C_x_1])


    #imagen de los puntos extendidos
    thin=np.zeros_like(thin)
    for i in range(0,len(sk_puntos)):
        thin[sk_puntos[i][0]][sk_puntos[i][1]]=255

    thin=cv2.bitwise_or(thin,aux_smooth_thin)
    #cv2.namedWindow("t",cv2.WINDOW_NORMAL)
    #cv2.imshow("t",thresh)
    #cv2.drawContours(thin,[contours[0]],0,255,1)
    #cv2.namedWindow("imgg",cv2.WINDOW_NORMAL)
    #v2.imshow("imgg",thin)
    #cv2.waitKey(0)
    return thin,contours


def mediaAxis(img,thresh):
    #blur = cv2.GaussianBlur(img,(3,3),0)
    #_,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    "Convert gray images to binary images using Otsu's method"
    BW_Original = img <= thresh    # must set object region as 1, background region as 0 !
    BW_Skeleton = zhangSuen(BW_Original)
    skeleto = img_as_ubyte(BW_Skeleton) #transformo skimage to opencv image
    #aa=np.array(BW_Skeleton,dtype=np.float64)#opencv image to skimage ???
    #opencv use numpy.uint8 type to represent binary images instead scikit-image numpy.float6
    return skeleto

def calcularEjeMedio(img):
    ret, thresh_ = cv2.threshold(img,254,255,cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    thresh_= cv2.morphologyEx(thresh_, cv2.MORPH_CLOSE,kernel)#elimino huecos
    thresh_= cv2.morphologyEx(thresh_, cv2.MORPH_OPEN, kernel)#elimino ruido del contorno del cromosoma
    thresh_=cv2.bitwise_not(extras.floodfill(cv2.bitwise_not(thresh_))) #rellenar huecos(si quedo alguno)
    thresh_=extras.eliminar_objetos_pequeños(thresh_,20)
    skeleto=mediaAxis(img.copy(),cv2.bitwise_not(thresh_))

    return skeleto,thresh_

def smoothEjeMedio(thin,thresh,step):
    ventana_extend=10#cantidad de puntos a tomar entre los puntos para extender la recta(mas distancia entre puntos suaviza mas)
    pruned_thin=rb.remove_branchs(thin,False)
    pruned_thin=img_as_ubyte(pruned_thin)
    if (len(np.transpose(np.nonzero(thin)))>=4):#si thin tiene 4 o mas --- puntos esto->sino
        pruned_thin[pruned_thin==255]=1
        smooth_thin=smooth.smoothing_thin(pruned_thin,step)
        aux_smooth_thin=smooth_thin.copy()
        aux_smooth_thin[aux_smooth_thin==255]=1
        smooth_thin_extend,cnt=extend_thin(aux_smooth_thin.copy(),thresh.copy(),ventana_extend)
    else: #sino -> forzar-> trazo la linea media vertical de la imagen
        thin=np.zeros_like(thin)
        h,w=thin.shape
        cv2.line(thin,(int(round(w/2)),0),(int(round(w/2)),h),255,1)
        _, cnt, _ = cv2.findContours(cv2.bitwise_not(thresh),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        smooth_thin_extend=cv2.bitwise_and(thin,cv2.bitwise_not(thresh))
        #cv2.namedWindow("imgs",cv2.WINDOW_NORMAL)
        #cv2.imshow("imgs",thin)
        #cv2.waitKey(0)


    return smooth_thin_extend,cnt

def relacion_aspecto(thresh):
    _, contours, _ = cv2.findContours(cv2.bitwise_not(thresh),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h = cv2.boundingRect(contours[0])
    aspect_ratio = float(w)/h
    return aspect_ratio
