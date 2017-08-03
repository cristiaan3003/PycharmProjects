Extracción de Caracteristicas
==========

Extraccion de caracteristicas de los cromosomas de forma ESTANDAR, mediante tecnicas de PROCESAMIENTO DE IMAGENES.

Este fichero contiene los  scripts que uso para la extraccion de caracteristicas.

WorkFlow
--------------------
Primero preprocesar las imagenes indivicuales de cromosomas para enderezar los curvos (Ejecutar: enderezarCromosomasCurvos.py). El script procesa todas las imagenes, contengan estas cromosomas curvo o no, si el cromosoma esta recto o casi recto la salida es el mismo cromosoma.Esto se realiza ya que la extracción de caracteristicas cromosomas rectos es mas simple y evita ciertos vicios.
Debe entregarle el PATH a "content_folder" que es donde se  encuentran las imagenes de cromosomas sin procesar y el PATH "salida_folder" donde se guardaranlas imagenes de los cromosomas enderezados.
Posteriormente y una vez finalizado el procesamiento de todas las imagenes de entrada y haber obtenido las imagenes de los cromosomas enderezados pasamos a la segunda fase de proceso.  
Ahora debe ejecutar el scrip que realiza la extracción de caracteristicas sobre los cromosomas enderezados (Ejecutar: featuresGrupo1.py ), este sript extrae las caracteristicas de los cromosomas y las guarda en la carpeta "clases". Se le debe entregar el PATH la carpeta donde se encuentran los cromosomas enderezados "content_folder"  y "clases" el path a la carpeta donde se guardaran las caracteristicas extraidas. Las caracteristicas seran guardadas en archivos .csv saparadas por clases de cromosomas, cada linea de los archivos representa las caracteristicas extraidas a un cromosoma. Se tiene 24 clases, por lo tanto se decidio crear 1(un) archivo por cada clase. 
Las caracteristicas extraidas deben ingresarse a un clasificador y entrenarlo. Pero es algo separado, que no explicare. Aqui solo extraigo las caracteristicas, las puede usar para realizar pruebas con distintos clasificadores. 

Archivos
--------------------

Así se crean secciones y subsecciones. Para crear una enumeración utilizamos:
+ pki-3_612.lis.txt : contiene el listado de nombre de los archivos que voy a prosesar, es cargado por los script
+ featuresGrupo1.py: algoritmo general que extrae las caracteristicas de los cromosomas ( Se extraen 128 caracteristicas por cada cromosoma.
El vector de caracteriticas se encuentra conformado de la siguiente manera:
los primeros 12 valores son el area cromosoma,largo del eje medio, indice centromerico, cantidad de bandas oscuras ,....,..
los siguiente valores hasta completar los 128 es el prefil de bandas. Los perfiles de bandas de todos los cromosomas se remuestrean de manera que todos tengan la misma longitud.
  Antes de lograr poder extraer las caracteristicas, se deben realizar distintos procesos sobre el cromosoma para individualizarlo. PAra obtener el eje medio se debe sortear problemas, como por ejemplo que el algoritmo de thinning deja brasos que no pertenecen al eje medio y se deben eliminar, y este eje medio se debe suavizar para que no tenga cambios bruzcos.  
Para lograr el algoritmo final que haga todo el proceso tome ideas de distintos papers de la tematica.



+ enderezarCromosomasCurvos.py: algoritmo general para enderezar cromosomas. Basado en: doi: 10.1016/j.patrec.2008.01.029
+ calcularAreasPerimetros.py: calcula las areas y perimetros de los cromosomas
+ centromere_index.py: calcular indices centromericos
+ densityProfile.py: distintios metodos referentes a perfil de densidad
+ extras.py: metodos extras, ejemplo: de carga de dataset
+ mediaAxisTransform.py: obtine el eje medio
+ order_points.py: basado en http://dip4fish.blogspot.com.ar/2011/06/open-curve-to-ordered-pixels-second.html
+ removeBRanchs.py: basado en http://dip4fish.blogspot.com.ar/2013/06/which-structuring-elements-to-detect.html
+ smoothing_thin.py: suaviza el eje medio obtenido 
+ featuresGrupo1.py: algoritmo general para extraccion de caracteristicas  


Carpetas
-----------------------
**single\_clase**: en esta carpeta estan las imagenes de los cromosomas individuales sin procesar que extraje de la base de datos. Adjunto esta carpeta en el repositorio para que puedan realizr test

**single\_clases\_enderezados**: esta carpeta la creo (crearla vacia) antes de ejecutar featuresGrupo1.py para que dentro de ella guarde los cromosomas enderezados. La adjunto al repo para que puedan realizar las pruebas de extracción de caracteristicas sin tener que primero procesar los cromosomas y enderazarlos. De esta forma ya tienen los cromosomas rectos. Si desean borrarla y procesar los cromosomas primero con enderezarCromosomasCurvos.py para realizar test lo pueden hacer. Incluso pueden cambiarle el nombre ( recordar cambiar tambien el nombre en la implementación si hacen esto).

**clases**: esta carpeta no la incluyo en el repo, por que es el resultado de las caracteristicas que extrae featuresGrupo1.py, pueden generar su contenido ejecutando dicho algoritmo. Deben crear la carpeta vacia antes de ejecutarlo, al finalizar dentro tendra las caracteristicas extraidas de los cromosomas.

###### **Webs consultadas**

- http://dip4fish.blogspot.com.ar

- **Base de datos**: http://www.fim.uni-passau.de/en/faculty/former-professors/mathematical-stochastics/chromosome-image-data (de esta base se datos se recortaron las images de los  cromosomas individuales para poder procesarlas)

###### Software, libs,frammworks, etc y todo eso que use para que esto ande

- SO: Ubuntu 16.04
- Codigo: Python 
- Version Python: 3.6
- Editor: PyCharm Edu 3.5
- scikit-image (0.12.3)
- scikit-learn (0.18.1)
- scipy (0.18.1)
- numpy (1.11.3)
- pandas (0.19.2)
- OpenCV 3.2
