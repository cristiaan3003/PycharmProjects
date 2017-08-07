import numpy as np
import csv
features = []
X_test=[]
X_test.append([1,2,3])
X_test.append([4,5,6])
X_test.append([7,8,9])
X_test.append([10,11,12])
Y_test=[]
Y_test.append(1)
Y_test.append(2)
Y_test.append(3)
Y_test.append(4)


myfile = open("features.csv", 'a')
for i in range(len(X_test)):
        #Tomar una imagen del Set de Test y obtener su salida en la dense_layer
        instance = X_test[i]
        instance = np.array(instance)
        wr = csv.writer(myfile, quoting=csv.QUOTE_MINIMAL)
        instance=instance.tolist()
        instance.append(Y_test[i])
        wr.writerow(instance)
myfile.close()

a=[[1, 2, 3, 1],
[4, 5, 6, 2],
[7, 8, 9, 3],
[10, 11, 12, 4]]
b=np.transpose(a)
c=np.transpose(b[:-1])
d=b[-1]
print(a)
print(b)
print(c)
print(d)

