import numpy as np
import matplotlib.pyplot as plt
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import pandas
import pickle
from sklearn import metrics

def read_csv_files(folder, Nfeatures=2500):
    '''
    Load all the "CSV" files in a folder, and store each one in a dictionary.
    '''

    clase = dict()

    for n in range(0,24):

      clase[n] = pandas.read_csv(folder +'clase' + str(n) + '.csv', sep=',',header=None, names=['f'+str(m) for m in range(0,Nfeatures)])

    return clase




if __name__ == "__main__":
    n_epoca_list = [1,5,10,20,30,40,50]
    '''clase = read_csv_files('clases/') #imagenes de 50x50px puestas el valor de sus pixel en archivo


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


    #Cargo Indices del particionado K-fold previamente realizado
    #de esta manera siempre puso los mismo datos para entrenamiento y test y puedo comparar
    filehandler = open('../Particionado_K-Fold/kfold.pki', 'rb')
    k_fold_indexs = pickle.load(filehandler)
    filehandler.close()

    X=np.array(X)
    Y=np.array(Y)
    train_prec =  []
    eval_prec = []



    for epoca in n_epoca_list:
        print("---")
        print(epoca)
        #print("TRAIN:", k_fold_indexs[fold][0], "TEST:", k_fold_indexs[fold][1])
        X_train, X_test = X[k_fold_indexs[0][0]], X[k_fold_indexs[0][1]]
        Y_train, Y_test = Y[k_fold_indexs[0][0]], Y[k_fold_indexs[0][1]]

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
        max_epochs=epoca,
        verbose=1,
        )


        # Train the network
        net1.fit(X_train, y_train)

        Y_pred1 = net1.predict(X_test)
        cm1=metrics.confusion_matrix(Y_test,Y_pred1)
        M1 = np.float64(cm1)
        uar1 = np.diag(M1)/np.sum(M1,1)

        Y_pred2 = net1.predict(X_train)
        cm2=metrics.confusion_matrix(Y_train,Y_pred2)
        M2 = np.float64(cm2)
        uar2 = np.diag(M2)/np.sum(M2,1)

        train_prec.append( np.mean(uar2))
        eval_prec.append(np.mean(uar1))

    print(train_prec)
    print(eval_prec)'''

    # Accuracy - n_epoca_list = [1,5,10,20,30,40,50]
    #train_prec=[0.7113100808166738, 0.9361973628243301, 0.9484900042535092, 0.9736282433007231, 0.9769034453424075, 0.9818375159506593, 0.9843045512547852]
    #eval_prec=[0.7079984691924991, 0.9207807118254879, 0.9253731343283582, 0.9456563337160352, 0.9479525449674704, 0.9402985074626866, 0.9491006505931879]

    #UAR - n_epoca_list = [1,5,10,20,30,40,50]
    train_prec=[0.68916770088857104, 0.92361482834636233, 0.95070723200848517, 0.96378659828350155, 0.97411783167093546, 0.98220647642288039, 0.98288384792933581]
    eval_prec=[0.66942993250046945, 0.89850017949196259, 0.93121758146172107, 0.93135848438634594, 0.93819228281009492, 0.94569588933246118, 0.93461729460844612]


     # graficar los resultados.
    #plt.plot(lista3, marker='o', linestyle='--', color='r', label = "Marzo")
    plt.plot(n_epoca_list, train_prec, marker='o',color='r', label='entrenamiento')
    plt.plot(n_epoca_list, eval_prec, marker='x',color='b', label='evaluación')
    plt.title('Gráfico de ajuste CNN')
    plt.legend(loc=4)
    plt.axis([1,50,0.5,1.01])
    #indice = np.arange(len(n_epoca_list))   # Declara un array
    plt.xticks(n_epoca_list, ("1", "5", "10","20","30","40","50"))
    #plt.yticks(np.arange(0.5,1.1,0.1))
    plt.ylabel('UAR')
    plt.xlabel('Número de Época')
    plt.show()


