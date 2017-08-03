import cv2
import numpy as np
import copy
import extras

#calcular Areas y Perimetros de todos los cromosomas de la clase
def calcular_areasCromo_perimsContornoCromo(string_path_img):
    perimetros=[]
    areas=[]
    from joblib import Parallel, delayed
    import multiprocessing
    num_cores = multiprocessing.cpu_count()
    for j in range(0,len(string_path_img)):#j->clase
        areaX=[]
        perimetroX=[]
        #results = Parallel(n_jobs=num_cores)(delayed(calcular_area_perimetro)(string_path_img[j][i]) for i in range(0,len(string_path_img[j])))
        for i in range(0,len(string_path_img[j])):#i ->img dentro de la clase
            full_path_img_j_i=string_path_img[j][i]
            area_cromosoma,perimetro_cromosoma,imagen_contorno=calcular_area_perimetro(full_path_img_j_i)
            areaX.append(area_cromosoma)
            perimetroX.append(perimetro_cromosoma)
        areas.append(areaX)
        perimetros.append(perimetroX)
    return areas,perimetros

#calcular area y perimetro de un cromosoma
def calcular_area_perimetro(full_path_img_j_i):
        img=extras.load_image(full_path_img_j_i)#img -> gray
        gray=cv2.copyMakeBorder(img.copy(), top=1, bottom=1, left=1, right=1, borderType= cv2.BORDER_CONSTANT, value=[255,255,255] )
        ret, thresh = cv2.threshold(gray,250,255,cv2.THRESH_BINARY)
        thresh=extras.eliminar_objetos_pequeños(thresh,20) #eliminar objetos pequeños
        thresh=extras.floodfill(cv2.bitwise_not(thresh)) #rellenar huecos
        area_cromosoma,perimetro_cromosoma,imagen_contorno=contorno(thresh)
        return area_cromosoma,perimetro_cromosoma,imagen_contorno

#Obtine informacion de Area y perimetro de un cromosoma
#Recibe como parametro el imagen binaria del cromosoma
def contorno(thresh):
    _, contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    imagen_contorno= np.zeros(thresh.shape,np.uint8)
    cv2.drawContours(imagen_contorno, contours, -1, 255, 1)
    perimetro_cromosoma = cv2.arcLength(contours[0],True)#perimetro del contorno
    moments = cv2.moments(contours[0])
    area_cromosoma = moments['m00'] ##area del cromosoma --> igual a area = cv2.contourArea(cnt)
    return area_cromosoma,perimetro_cromosoma,imagen_contorno

#normaliza vector (multidimensional) de perimetro y  area
#normalizar [MAX MIN] to [0,1]
def normalizar_vector(vector):
    vector_normalizado=copy.deepcopy(vector)
    import operator
    flat_list=[item for sublist in vector for item in sublist]
    min_index, min_value = min(enumerate(flat_list), key=operator.itemgetter(1))
    max_index, max_value = max(enumerate(flat_list), key=operator.itemgetter(1))
    #print(flat_list)
    for i in range(0,len(vector)):
        for j in range(0,len(vector[i])):
            #print(vector_normalizado[i][j])
            if max_value!=min_value:
                vector_normalizado[i][j]=((vector[i][j]-min_value)/(max_value-min_value))
            else:
                vector_normalizado.append(1)
    return vector_normalizado
