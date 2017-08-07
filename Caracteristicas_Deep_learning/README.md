Estan son las caracteristicas extraidas por la CNN. El contenido de esta carpeta se pega desde las caracteristicas que extrajo la red. 
RECORDAR: si se quiere actualizar el juego de caracteristicas K-FOLD extradias sobre un conjunto de imagenes
(realizar las particiones primero-es este esta k=10, si de desea cambiar se debe reparticionar)

1. Las imagenes tiene que estar volcadas a un vector que contiene la informacion de intensidad->dump_picked_objet.py
2. Ejecutar cnn\_extraction\_features\_k\_fold.py -> guardara las caracteristicas extraidas en ->PycharmProjects/dl\_cnn\_features/Caracteristicas\_Deep\_learning/featuresTestXXX

3. Copiar la carpeta  Caracteristicas\_Deep\_learning ---> PycharmProjects ( es de donde leeran su contenido los clasificadores)
