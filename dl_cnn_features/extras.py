import os
import numpy as np
import math
import cv2
from scipy import ndimage
from skimage import measure


#count files and folder
def count_folders_files(rel_path):
    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    abs_file_path = os.path.join(script_dir, rel_path)
    files = folders = 0
    for _, dirnames, filenames in os.walk(abs_file_path):
        files += len(filenames)
        folders += len(dirnames)
    return folders,files

#load imagen
def load_image(full_path):
    img = cv2.imread(full_path)
    if len(img.shape)==3:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img

#build rel_path
#rel_path->ejem:/single_clase/9700TEST.6.tiff/clase1/clase1_0_9700TEST.6.tiff
#retorna los path a los archivos de la clase que se la pase en clase_folder
def build_rel_path(content_folder,name_example_folder,clase_folder):
    rel_path=content_folder+"/"+name_example_folder+"/"+clase_folder
    num_files_folder=count_folders_files(rel_path)[1]
    rel_paths_img_clase=np.empty(num_files_folder, dtype='object')
    for i in range(0,len(rel_paths_img_clase)):
        rel_paths_img_clase[i]=rel_path+"/"+clase_folder+"_"+str(i)+"_"+name_example_folder
    return rel_paths_img_clase

#los ALL PATH del caso de estudio (path a las imagenes separadas por clase)
def load_all_PATH(content_folder,name_example_folder):
    path=content_folder+"/"+name_example_folder
    cantidad_de_clases=count_folders_files(path)[0]
    string_paths_img=np.empty(cantidad_de_clases, dtype='object')
    for j in range(0,cantidad_de_clases):
        clase_folder="clase"+str(j+1)
        rel_paths_img_clase=build_rel_path(content_folder,name_example_folder,clase_folder)
        string_paths_img[j]=rel_paths_img_clase
    return string_paths_img

#Cargar nombres de las carpetas que contienen cada caso de estudio (names examples folders)
#los nombres de todas las carpetas estan un un archivo de texto
def cargar_nombres(filepath):
    archivo = open(filepath, "r")
    leer_fila= archivo.readlines()
    archivo.close()
    a=[]
    for lista in leer_fila:
        # revisamos si tiene un salto de linea al final para quitarlselo.
        if lista[-1]=="\n":
            a.append(lista[:-1].split(", ")[0])
        else:
            dato=lista.split(", ")
    archivo.close()
    return a

#=====================================================================================
#=====================================================================================
#Buscar indices FILA para cortar la parte blanca de la imagen ROI
def findStartIndex(img,h,w):
    for i in range(0, h):#para fila i
           aux=img[i:i+1,0:w]
           sum=math.fsum(aux[0])
           if 255*w!=sum:
                return i-1

def findEndIndex(img,h,w):
    for i in reversed(range(0, h)):#para fila i
           aux=img[i:i+1,0:w]
           sum=math.fsum(aux[0])
           if 255*w!=sum:
                return i+1

def findStartEndFila(img):
    h,w=img.shape
    index1=findStartIndex(img,h,w)
    index2=findEndIndex(img,h,w)
    return index1,index2

#=====================================================================================
#=====================================================================================
#Buscar indices COLUMNA para cortar la parte blanca de la imagen ROI
def findStartIndexCol(img,h,w):
    for i in range(0, w):
           aux=img[0:h,i:i+1]
           sum=math.fsum(np.transpose(aux)[0])
           if 255*h!=sum:
                return i-1

def findEndIndexCol(img,h,w):
    for i in reversed(range(0, w)):
           aux=img[0:h,i:i+1]
           sum=math.fsum(np.transpose(aux)[0])
           if 255*h!=sum:
                return i+1

def findStartEndColumna(img):
    h,w=img.shape
    index1=findStartIndexCol(img,h,w)
    index2=findEndIndexCol(img,h,w)
    return index1,index2

#=====================================================================================
#=====================================================================================


#ELIMINA OBJETOS DE TAMAÑO menos a "tamano" EN LA IMAGEN
def eliminar_objetos_pequeños(th,tamano):
    blobs_labels = measure.label(th, background=255)
    boxes_objetos= ndimage.find_objects(blobs_labels)
    #print(boxes_objetos[0][0])
    #ejemplo1
    #(slice(0, 2, None), slice(19, 23, None))
    #So this object is found at x = 19–23 and y = 0–2
    if len(boxes_objetos)>1:#solo debe haber 1 objeto->el cromosoma
        #filtro objetos muy pequeños ya que sob objetos indeseados (no cromosomas)
        aux_boxes=[]
        for l in range(0,len(boxes_objetos)):
            if len(np.transpose(np.nonzero(cv2.bitwise_not(th[boxes_objetos[l]]))))<tamano:
                th[boxes_objetos[l]]=255
    return th

#RELLENA HUECOS
def floodfill(th):
    # Copy the thresholded image.
    im_floodfill = th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = th | im_floodfill_inv

    return im_out
