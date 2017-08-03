import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
import pandas
from random import shuffle
import math
import pickle as pk

from sklearn import svm
from sklearn import datasets
from sklearn import preprocessing

from sklearn import model_selection # <---- PARTICIONADO


from sklearn import metrics
from sklearn.metrics import confusion_matrix


def read_csv_files(folder, Nfeatures=102):
    '''
    Load all the "CSV" files in a folder, and store each one in a dictionary.
    '''

    #filenames = glob.glob(rel_path + '*.csv')

    clase = dict()

    for n in range(0,24):

      clase[n] = pandas.read_csv(folder +'clase' + str(n) + '.csv', sep=',',header=None, names=['f'+str(m) for m in range(0,Nfeatures)])
#        clase[n+1] = np.array(pandas.read_csv(folder +'clase' + str(n+1) + '.csv', sep=',',header=None, names=['f'+str(m) for m in range(0,Nfeatures)]))
    return clase




def k_fold_cross_validation(X, K, randomise = False):
    k_fold= dict()
    n,m=X.shape
    size = math.ceil(n/float(K))
    indices=list(range(n))
    #print(n,m)
    #print(size)
    #print(indices)
    if randomise:
        shuffle(indices)
    #print(indices)
    count=0
    test=0
    for k in range(0,n,size):
        #print("-----_____-------____-----")
        if (count+1)!=K:
            #print("rango:"+str(k+count)+"-"+str(k+size+count))
            test=indices[k+count:k+size+count]
            inds_to_delete=list(range(k+count,k+size+count))
            inds_to_delete = sorted(inds_to_delete, reverse=True) # [5,3,1]

            train=indices.copy()
            for i in inds_to_delete:  # iterate in order
                train = np.delete(train,i)
        else:
            #print("rango:"+str(k+count)+"-"+str(n))
            test=indices[k+count:n]
            inds_to_delete=list(range(k+count,n))
            inds_to_delete = sorted(inds_to_delete, reverse=True) # [5,3,1]

            train=indices.copy()
            for i in inds_to_delete:  # iterate in order
                train = np.delete(train,i)
        k_fold[count]= train,test
        #print(train)
        #print(len(train))
        #print(test)
        #print(len(test))
        #print(count)
        count=count+1
    return k_fold




    #training = [x for i, x in enumerate(X) if i % K != k]
    #validation = [x for i, x in enumerate(X) if i % K == k]
    #yield training, validation



#===================================================
#==========MAIN========================
#===================================================
if __name__ == "__main__":
    clase = read_csv_files('../extraccionCaracteristicas/clases/',128)
    print(clase[0])
    print(clase[1].shape)
    #print(clase[2].shape)
    #print(clase[23].shape)
    clases_a_usar = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

    X = []
    Y = []

    for idx in clases_a_usar:

        x = np.array(clase[idx]).tolist()
        y = idx * np.ones((clase[idx].shape[0]))
        y = y.tolist()

        X.extend(x)
        Y.extend(y)

    X=np.array(X)
    Y=np.array(Y)

    k_fold=k_fold_cross_validation(X, K=10,randomise = True)

    file_pi = open('kfold.pki', 'wb')
    pk.dump(k_fold, file_pi)
    file_pi.close()


    '''kf = model_selection.StratifiedKFold(n_splits=10,shuffle=True, random_state=42)
    print(kf)

    k_fold= dict()
    print(kf)
    count=0
    for train_index, test_index in kf.split(X,Y):
        #print("---")
        #print("TRAIN:", train_index, "TEST:", test_index)
        k_fold[count]= train_index,test_index
        count=count+1


    file_pi = open('kfold.pki', 'wb')
    pk.dump(k_fold, file_pi)
    file_pi.close()'''




