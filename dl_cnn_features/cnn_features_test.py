#Test de la CNN

#usado para test de la cnn sin particionado kfold - test de rendimiento preliminar antes de hacer k-fold
#graficado de los filtros en etapas intermedias
from theano import function, config, shared, tensor
import numpy
import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from theano import tensor as T
#from urllib import urlretrieve
from urllib.request import urlretrieve
import os
import gzip
import numpy as np
import theano
import lasagne
from lasagne import layers
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
        verbose=1,
        )



    # Train the network
    net1.fit(X_train, y_train)
    pickle.dump( net1, open( "net1.plk", "wb" ) )'''

    net1=pickle.load(open( "net1.plk", "rb" ))
    #Prediction and Confusion Matrix
    #preds = net1.predict(X_test)

    #MATRIZ DE CONFUSION
    #cm = confusion_matrix(y_test, preds)
    #plt.matshow(cm)
    #plt.title('Confusion matrix')
    #plt.colorbar()
    #plt.ylabel('True label')
    #plt.xlabel('Predicted label')
    #plt.show()


    #Filters Visualization

              #    ('conv2d1', layers.Conv2DLayer),
             #   ('maxpool1', layers.MaxPool2DLayer),
             #   ('conv2d2', layers.Conv2DLayer),
             #   ('maxpool2', layers.MaxPool2DLayer),
             #   ('dropout1', layers.DropoutLayer),
             #   ('dense', layers.DenseLayer),
             #   ('dropout2', layers.DropoutLayer),
             #   ('output'

    #Theano layer functions and Feature Extraction
    dense_layer = layers.get_output(net1.layers_['dense'], deterministic=True)
    output_layer = layers.get_output(net1.layers_['output'], deterministic=True)
    input_var = net1.layers_['input'].input_var
    f_output = theano.function([input_var], output_layer)
    f_dense = theano.function([input_var], dense_layer)


    # #-----
    # import cv2
    # img=extras.load_image("/home/asusn56/PycharmProjects/dl_cnn_features/single_clase_enderezados/97002126.18.tiff/clase2/clase2_0_97002126.18.tiff")
    # img = cv2.resize(img, (50, 50))
    # # #print(img.shape)
    # arr = np.array(img)
    # # # record the original shape
    # shape = arr.shape
    # # # make a 1-dimensional view of arr
    # flat_arr = arr.ravel()
    # # #Normalized Data
    # flat_arr_norm=(flat_arr-min(flat_arr))/(max(flat_arr)-min(flat_arr))
    # flat_arr_norm[flat_arr_norm==1.]=0.
    # flat_arr_norm=np.array(flat_arr_norm)
    # flat_arr_norm=flat_arr_norm.astype(np.float32)
    # test = flat_arr_norm.reshape((-1, 1, 50, 50))
    # instance=test[0][None, :, :]
    # print(instance)

    #Tomar una imagen del Set de Test y probar la salida
    instance = X_test[18][None, :, :]

    #print(instance)

    #visualize.plot_loss(net1)

    #visualize.plot_conv_weights(net1.layers_['conv2d1'])

    #visualize.plot_conv_activity(net1.layers_['conv2d1'],instance)
    #visualize.plot_conv_activity(net1.layers_['maxpool1'],instance)

    visualize.plot_conv_weights(net1.layers_['conv2d2'])

    #visualize.plot_conv_activity(net1.layers_['conv2d2'],instance)
    #visualize.plot_conv_activity(net1.layers_['maxpool2'],instance)




                #('conv2d1'
                #('maxpool1', layers.MaxPool2DLayer),
                #('conv2d2', layers.Conv2DLayer),
                #('maxpool2', layers.MaxPool2DLayer),
                #('dropout1', layers.DropoutLayer),
                #('dense', layers.DenseLayer),
                #('dropout2', layers.DropoutLayer),
                #('output', layers.DenseLayer),
    #visualize.plot_conv_weights(net1.layers_['conv2d2'])
    #visualize.plot_conv_activity(net1.layers_['conv2d2'],instance)




    #print(instance)
    pred = f_output(instance)
    N = pred.shape[1]
    #PLot el resultado
    plt.figure(2)
    plt.title('Pred tst')
    plt.bar(range(N), pred.ravel())
    plt.show()



    #Tomar la activacion de la capa anterior a la salida
    #puedo usar estas como caracteristicas?-> pasar esto a un svm para determinar la salida final?
    #o conviene usar f_output directamente?
    #pred = f_dense(instance)
    #N = pred.shape[1]

    #plt.figure(3)
    #plt.title('dense layer')
    #plt.bar(range(N), pred.ravel())
    #plt.show()



    '''

    #______________________________________________________
    #-------------------------------------------------------
    #-------------------------------------------------------
    #-------------------------------------------------------
    #-------------------------------------------------------
    #-------------------------------------------------------
    #-------------------------------------------------------
    #-------------------------------------------------------
    #-------------------------------------------------------
    #-------------------------------------------------------
    net1=pickle.load(open( "net1.plk", "rb" ))
    #Prediction and Confusion Matrix
    preds = net1.predict(X_test)
    #======================
    # SCORE CALCULATION
    #======================
    score = net1.score(X_test, Y_test)



    target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7',
                    'class 8', 'class 9', 'class 10', 'class 11', 'class 12', 'class 13', 'class 14',
                    'class 15', 'class 16', 'class 17', 'class 18', 'class 19', 'class 20', 'class 21',
                    'class 22', 'class 23']
    print("classification_report")
    print(metrics.classification_report(Y_test, preds, target_names=target_names))



    #===================
    # CONFUSSION MATRIX
    #===================
    print('\nCONFUSSION MATRIX')
    cm=metrics.confusion_matrix(Y_test,preds)
    plt.matshow(cm)
    plt.title('Matriz de confusión')
    plt.colorbar()
    plt.ylabel('Valor real')
    plt.xlabel('Valor predicho')
    plt.show()'''
