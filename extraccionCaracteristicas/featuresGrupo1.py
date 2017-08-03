import extras
import os
import cv2
import mediaAxisTransform
import densityProfile as dp
import normalizar_Clase_i as normalizar
import numpy as np
import csv
import centromere_index as ci
import matplotlib.pyplot as plt
from skimage import img_as_ubyte

def flatList(Clase_i):
     nor=[]
     for j in range(0,len(Clase_i)):
         ss=[]
         for i in range(0,int(len(Clase_i[j])/13)):
             aux=Clase_i[j][13*i:13*(i+1)]
             ff=[aux[0]]+[aux[1]]+[aux[2]]+[aux[3]]+[aux[4]]+[aux[5]]+[aux[6]]+[aux[7]]+[aux[8]]+[aux[9]]+[aux[10]]+[aux[11]]+aux[12]
             ss.append(ff)
         nor.append(ss)
     return nor

#Procesa Todos los cromosomas de una carpeta
def procesarCarpeta(content_folder,name_example_folder,border,step,sampleDensity,normaliza=0):
    print(name_example_folder)
    string_path_img=extras.load_all_PATH(content_folder,name_example_folder)
    count=0

    cLase_i=[]
    for j in range(0,len(string_path_img)):#j->clase
        Chromo_i=[]
        for i in range(0,len(string_path_img[j])):#i ->img dentro de la clase

            #print(string_path_img[j][i])
            img=extras.load_image(string_path_img[j][i])
            img=cv2.copyMakeBorder(img, top=border, bottom=border, left=border, right=border, borderType= cv2.BORDER_CONSTANT, value=255 )

            #....
            #Calcular aqui las caracteristicas para esta imagen
                #1)CALCULAR EJE MEDIO DE LA IMAGEN
            thin,thresh=mediaAxisTransform.calcularEjeMedio(img)
            smooth_thin_extend,contours=mediaAxisTransform.smoothEjeMedio(thin,thresh,step)
                #2) DENSITY PROFILE
            dP,centromere_pos=dp.dProfile_and_centromere_pos(img.copy(),smooth_thin_extend.copy(),thresh.copy())
                #3) CHROMOSOME AREA (en cantidad de pixeles)
            areaChromosome=len(np.transpose(np.nonzero(thresh)))
                #4) CHROMOSOME LENGTH (en cantidad de pixeles)
            lengthChromosome=len(np.transpose(np.nonzero(smooth_thin_extend)))
                #5)resample DENsity
            resampled_density=dp.re_sampled(dP,sampleDensity)
            #areasChromos.append(areaChromosome)
                #6) CI length -> centromere index (CI) expesed in ratio of the length
            CI_length= ci.centromere_index_ratio_of_the_length(smooth_thin_extend.copy(),centromere_pos)
                #7) CI area -> centromere index (CI) expesed in ratio of the area
            CI_area= ci.centromere_index_ratio_of_the_area(thresh.copy(),centromere_pos)
                #8) Intensity gradient determination - gradient profile
            gP=dp.gProfile(dP)
                #9) sign profile
            signP=np.sign(gP)
                #10) the number of bands
            numBands_p_arm,numBands_q_arm=dp.numBands_p_arm_q_arm(signP.copy(),centromere_pos)
                #11) total bands
            total_bands=numBands_p_arm+numBands_q_arm
                #12) aspect ratio
            aspect_ratio=mediaAxisTransform.relacion_aspecto(thresh)
                #13)the total number of black bands in the chromosome
            numBandsBlack,_=dp.numBandsBlackWhite(signP.copy())
                #14) the total number of white bands in the chromosome
            numBandsWhite=total_bands-numBandsBlack
                #15) the number of black band un q-arm
            numBandsBlack_q_arm,_=dp.numBandsBlack_q_arm(signP.copy(),centromere_pos)
                #16) the number of white band un q-arm
            numBandsWhite_q_arm=numBands_q_arm-numBandsBlack_q_arm




            Chromo_i.append(areaChromosome)
            Chromo_i.append(lengthChromosome)
            Chromo_i.append(CI_length)
            Chromo_i.append(CI_area)
            Chromo_i.append(numBands_p_arm)
            Chromo_i.append(numBands_q_arm)
            Chromo_i.append(total_bands)
            Chromo_i.append(aspect_ratio)
            Chromo_i.append(numBandsBlack)
            Chromo_i.append(numBandsWhite)
            Chromo_i.append(numBandsBlack_q_arm)
            Chromo_i.append(numBandsWhite_q_arm)
            Chromo_i.append(list(resampled_density))
            #print(CI_area)
        #print("------------------")
        cLase_i.append(Chromo_i)
    #print(len(cLase_i[23]))
    if normaliza==1:
        nor=flatList(cLase_i)
        final=normalizar.normalizar_areas_length_band(nor)
    else:
        final=flatList(cLase_i)
        print(final)


    return final

def guardarCarpeta(folder,datos):
    for j in range(0,len(datos)):
        for i in range(0,len(datos[j])):
             myfile = open(folder+"/clase"+str(j)+".csv", 'a')#abrir y agregar al final si existe- sino lo crea y agrega
             wr = csv.writer(myfile, quoting=csv.QUOTE_MINIMAL)
             wr.writerow(datos[j][i])
             myfile.close()



#======================================================================
#======================================================================
if __name__ == "__main__":

        content_folder="single_clase_enderezados" #path a la carpeta donde se encuentran las imagenes de cromosomas
        clases="clases" #path a la carpeta donde se guardaran las caracteristicas extraidas
        script_dir = os.path.dirname(__file__)
        abs_file_path = os.path.join(script_dir, content_folder)
        lista_name_example_folder=[name for name in os.listdir(abs_file_path) ]

        #========================================================================
        step=10#STEP OF SMOOTH LINE
        border=2
        sampleDensity=116 #resampled_density of N(cantidad de caracteristicas perfil de bandas)
        normalizar=0
        #os.makedirs(clases, mode=0o777)
        for i in range(0,len(lista_name_example_folder)):
            print(i)
            salida=procesarCarpeta(content_folder,lista_name_example_folder[i],border,step,sampleDensity)
            guardarCarpeta(clases,salida)
