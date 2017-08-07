#extracccion de caracteristicas DEEP LEARNING sin k-fold
# Extracción de caracteristicas test con una sola particion para test del rendimiento preliminar de la CNN


from theano import function, config, shared, tensor
import numpy
import time
import matplotlib
import matplotlib.pyplot as plt
import csv
import matplotlib.cm as cm
from theano import tensor as T
#from urllib import urlretrieve
from urllib.request import urlretrieve
import os
import gzip
import numpy as np
from lasagne.nonlinearities import softmax
import theano
import lasagne
from lasagne import layers
from nolearn.lasagne import TrainSplit
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas
from sklearn import model_selection
import pickle
import sklearn
from sklearn import preprocessing
import extras
from sklearn import metrics

def read_csv_files(folder, Nfeatures=2500):
    '''
    Load all the "CSV" files in a folder, and store each one in a dictionary.
    '''

    #filenames = glob.glob(rel_path + '*.csv')

    clase = dict()

    for n in range(0,24):

      clase[n] = pandas.read_csv(folder +'clase' + str(n) + '.csv', sep=',',header=None, names=['f'+str(m) for m in range(0,Nfeatures)])
#        clase[n+1] = np.array(pandas.read_csv(folder +'clase' + str(n+1) + '.csv', sep=',',header=None, names=['f'+str(m) for m in range(0,Nfeatures)]))
    return clase




if __name__ == "__main__":
    clase = read_csv_files('clases/') #imagenes de 50x50px puestas el valor de sus pixel en archivo


    clases_a_usar = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

    X = []
    Y = []

    for idx in clases_a_usar:

        x = np.array(clase[idx]).tolist()
        y = idx * np.ones((clase[idx].shape[0]))
        y = y.tolist()

        X.extend(x)
        Y.extend(y)

    clase=[]

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y,test_size=0.2,random_state=0)

    X_train = np.array(X_train)
    X_train = X_train.astype(np.float32)

    X_test = np.array(X_test)
    X_test = X_test.astype(np.float32)
    #print(X_test)
    #print(Y_test)

    Y_train = np.array(Y_train)
    Y_train = Y_train.astype(np.float32)

    Y_test = np.array(Y_test)
    Y_test = Y_test.astype(np.float32)

    X_train = X_train.reshape((-1, 1, 50, 50))
    X_test = X_test.reshape((-1, 1, 50, 50))
    y_train = Y_train.astype(np.uint8)
    y_test = Y_test.astype(np.uint8)

    '''net1 = NeuralNet(
        layers=[('input', layers.InputLayer),
                ('conv2d1', layers.Conv2DLayer),
                ('maxpool1', layers.MaxPool2DLayer),
                ('conv2d2', layers.Conv2DLayer),
                ('maxpool2', layers.MaxPool2DLayer),
                ('conv3d3', layers.Conv2DLayer),
                ('maxpool3', layers.MaxPool2DLayer),
                ('dropout1', layers.DropoutLayer),
                ('dense', layers.DenseLayer),
                ('dropout2', layers.DropoutLayer),
                ('output', layers.DenseLayer),
                ],
        # input layer
        input_shape=(None, 1, 50, 50),
        # layer conv2d1
        conv2d1_num_filters=64,
        conv2d1_filter_size=(3, 3),
        conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
        conv2d1_W=lasagne.init.GlorotUniform(),
        # layer maxpool1
        maxpool1_pool_size=(2, 2),
        # layer conv2d2
        conv2d2_num_filters=64,
        conv2d2_filter_size=(3, 3),
        conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
        # layer maxpool2
        maxpool2_pool_size=(2, 2),
        # layer conv3d3
        conv3d3_num_filters=64,
        conv3d3_filter_size=(3, 3),
        conv3d3_nonlinearity=lasagne.nonlinearities.rectify,
        # layer maxpool3
        maxpool3_pool_size=(2, 2),

        # dropout1
        dropout1_p=0.5,
        # dense
        dense_num_units=128,
        dense_nonlinearity=lasagne.nonlinearities.rectify,
        # dropout2
        dropout2_p=0.5,
        # output
        output_nonlinearity=lasagne.nonlinearities.softmax,
        output_num_units=24,
        # optimization method params
        update=nesterov_momentum,
        update_learning_rate=0.02,
        update_momentum=0.9,
        max_epochs=35,
        verbose=2,
        )'''
    net1 = NeuralNet(
        layers=[('input', layers.InputLayer),
                ('conv2d1', layers.Conv2DLayer),
                ('maxpool1', layers.MaxPool2DLayer),
                ('conv2d2', layers.Conv2DLayer),
                ('maxpool2', layers.MaxPool2DLayer),
                ('dropout1', layers.DropoutLayer),
                ('dense', layers.DenseLayer),
                ('dropout2', layers.DropoutLayer),
                ('output', layers.DenseLayer),
                ],
        # input layer
        input_shape=(None, 1, 50, 50),
        # layer conv2d1
        conv2d1_num_filters=32,
        conv2d1_filter_size=(5, 5),
        conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
        conv2d1_W=lasagne.init.GlorotUniform(),
        # layer maxpool1
        maxpool1_pool_size=(2, 2),
        # layer conv2d2
        conv2d2_num_filters=32,
        conv2d2_filter_size=(5, 5),
        conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
        # layer maxpool2
        maxpool2_pool_size=(2, 2),
        # dropout1
        dropout1_p=0.5,
        # dense
        dense_num_units=128,
        dense_nonlinearity=lasagne.nonlinearities.rectify,
        # dropout2
        dropout2_p=0.5,
        # output
        output_nonlinearity=lasagne.nonlinearities.softmax,
        output_num_units=24,
        # optimization method params
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,
        max_epochs=50,
        verbose=2,
        )




    # Train the network
    net1.fit(X_train, y_train)

    #Theano layer functions and Feature Extraction
    dense_layer = layers.get_output(net1.layers_['dense'], deterministic=True)
    output_layer = layers.get_output(net1.layers_['output'], deterministic=True)
    input_var = net1.layers_['input'].input_var
    f_output = theano.function([input_var], output_layer)
    f_dense = theano.function([input_var], dense_layer)



    featuresTest = []
    myfile = open("featuresTest.csv", 'a')
    for i in range(len(X_test)):
        print(i)
        #Tomar una imagen del Set de Test y obtener su salida en la dense_layer
        instance = X_test[i][None, :, :]
        out_dense = f_dense(instance) #salida de la capa densa
        out_dense=out_dense[0]
        out_dense=out_dense.tolist()#de numpy a list
        out_dense.append(Y_test[i])#agrego y_true en el final
        wr = csv.writer(myfile, quoting=csv.QUOTE_MINIMAL) #escribo a archivo
        wr.writerow(out_dense)
    myfile.close()

    featuresTrain = []
    myfile = open("featuresTrain.csv", 'a')
    for i in range(len(X_train)):
        print(i)
        #Tomar una imagen del Set de Test y obtener su salida en la dense_layer
        instance = X_train[i][None, :, :]
        out_dense = f_dense(instance) #salida de la capa densa
        out_dense=out_dense[0]
        out_dense=out_dense.tolist()#de numpy a list
        out_dense.append(Y_train[i])#agrego y_true en el final
        wr = csv.writer(myfile, quoting=csv.QUOTE_MINIMAL) #escribo a archivo
        wr.writerow(out_dense)
    myfile.close()

