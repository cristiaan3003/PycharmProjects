#Script para guardar las imagenes de los cromosomas en el disco como
#un vector de numeros, el archivo generado luego es cargado por
#cnn_extraction_features_XXXX.py
#------------------------------------------------------------
#Esta codificado para que pueda leer cada una de la imagenes de la base de datos (cromosomas enderezados)
#para que lea otra base de datos debe ser modificado

import extras
import os
import _pickle as pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
from skimage import img_as_ubyte

#Procesa Todos los cromosomas de una carpeta
def procesarCarpeta(content_folder,name_example_folder):
    #print(name_example_folder)
    string_path_img=extras.load_all_PATH(content_folder,name_example_folder)

    count=0
    for j in range(0,len(string_path_img)):#j->clase
        for i in range(0,len(string_path_img[j])):#i ->img dentro de la clase
            print(string_path_img[j][i])
            img=extras.load_image(string_path_img[j][i])
            img = cv2.resize(img, (50, 50))
            #print(img.shape)
            arr = np.array(img)
            # record the original shape
            shape = arr.shape
            # make a 1-dimensional view of arr
            flat_arr = arr.ravel()
            #Normalized Data
            flat_arr_norm=(flat_arr-min(flat_arr))/(max(flat_arr)-min(flat_arr))
            flat_arr_norm[flat_arr_norm==1.]=0.
            # convert it to a matrix
            #vector = np.matrix(flat_arr_norm)
            # reform a numpy array of the original shape
            #arr2 = np.asarray(vector).reshape(shape)
            #arr2[arr2==1.]=0.
            #print(arr2)
            file = open("clases/clase"+str(count)+".csv", "a")
            wr = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
            wr.writerow(flat_arr_norm)
            file.close()

            #Chromo_i.append(arr2)
        count=count+1




#======================================================================
#======================================================================
if __name__ == "__main__":

        content_folder="../extraccionCaracteristicas/single_clase_enderezados" #carpeta donde estan los ejemplos (en este caso enderezados)
        clases="clases"#carpeta donde se guardara la informacion extraida ( en realidad se transforma la imagenes a vectores, los vectores guardan imagen . informacion de intensidad de cada pixel)
        script_dir = os.path.dirname(__file__)
        abs_file_path = os.path.join(script_dir, content_folder)
        lista_name_example_folder=[name for name in os.listdir(abs_file_path)]

        #========================================================================
        #Procesa todas las carpetas dentro del directorio "content_folder"
        for i in range(0,len(lista_name_example_folder)):
            print(i)
            procesarCarpeta(content_folder,lista_name_example_folder[i])


