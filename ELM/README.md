Clasificacion con ELM (Extreme Learning Machine)
----------------------------------

**Caracteristicas Estandar**


Clasificacion con ELM de las caracteristicas extraidas de los cromosomas mediante tecnicas de procesamiento de imagenes. Se hace particionado K-fold. El particionador realiza el particionado sobre los indices, particiona los indices y recorre luego los rango de indices.

El particionado es un paso previo a aplicar el clasificador

**clase**: PATH a la carpeta que contiene las caracteristicas de los cromosomas (archivos .csv)
**clases_a_usar**: clases que se van a entregar al clasificador para que realize su tarea. Por defecto las clases estan numeradas desde 0 a 23 ( 24 clases). Default se entregan todas las clases para que las clasifique, para realizar test pueden jugar con el numero de clases que se le da al clasificador.


Ejecutar ELM\_k\_fold.py
