import numpy as np
import pandas
from sklearn import model_selection
from sklearn import linear_model
from sklearn_extensions.extreme_learning_machines.elm import ELMClassifier
from sklearn import metrics
import matplotlib.pyplot as plt


def read_csv_files(folder, Nfeatures=102):
    '''
    Load all the "CSV" files in a folder, and store each one in a dictionary.
    '''

    clase = dict()

    for n in range(0,24):

      clase[n] = pandas.read_csv(folder +'clase' + str(n) + '.csv', sep=',',header=None, names=['f'+str(m) for m in range(0,Nfeatures)])
#        clase[n+1] = np.array(pandas.read_csv(folder +'clase' + str(n+1) + '.csv', sep=',',header=None, names=['f'+str(m) for m in range(0,Nfeatures)]))
    return clase


if __name__ == '__main__':

    clase = read_csv_files('../Caracteristicas_Estandar/clases/',128)
    clases_a_usar = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

    X = []
    Y = []

    for idx in clases_a_usar:

        x = np.array(clase[idx]).tolist()
        y = idx * np.ones((clase[idx].shape[0]))
        y = y.tolist()
        X.extend(x)
        Y.extend(y)

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y,test_size=0.2,random_state=0)

    # Ahora si convierto a NUMPY
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)

    from sklearn import preprocessing
     #----------------------------------------------
    # PREPROCESSING?? --> i.e. mean=0.0 - var=1.0
    #----------------------------------------------
    scaler = preprocessing.MinMaxScaler(feature_range=(0,1)).fit(X_train)  # Built scaler to scale values in range [Xmin,Xmax]
    # APPLY TRAINED SCALER
    sX_train = scaler.transform(X_train)

    sX_test = scaler.transform(X_test)

    sX_train.mean(axis=0)

    TRAIN = sX_train # --> Datos ESTANDARIZADOS

    TEST = sX_test # --> Datos ESTANDARIZADOS


    clf=ELMClassifier(n_hidden=4000,alpha=0.95,activation_func='multiquadric',activation_args=None,
                      user_components=None,regressor=linear_model.Ridge(),random_state=None)

    clf.fit(TRAIN, Y_train)

    #TEST = X_test  # --> Datos CRUDOS
    TEST = sX_test # --> Datos ESTANDARIZADOS
    #TEST = nX_test # --> Datos NORMALIZADOS

    Y_pred = clf.predict(TEST)

    #======================
    # SCORE CALCULATION
    #======================
    score = clf.score(TEST, Y_test)

    print("\nAccuracy: %0.2f" % (score))

    #Cohen’s kappa-[-1,1]->Si>0.8 se considera ok
    score=metrics.cohen_kappa_score(Y_test, Y_pred)
    print("Cohen’s kappa: %0.2f " % score)
    # Hamming loss¶ . when 0 excelent
    score=metrics.hamming_loss(Y_test, Y_pred)
    print("Hamming loss: %0.2f " % score)

    #======================
    #cross_val_score
    #======================
    #['accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'f1_macro',
    # 'f1_micro', 'f1_samples', 'f1_weighted', 'neg_log_loss', 'neg_mean_absolute_error',
    # 'neg_mean_squared_error', 'neg_median_absolute_error', 'precision', 'precision_macro',
    # 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'recall', 'recall_macro',
    # 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc']
    print("Cross-Val")
    #accuracy
    scores = model_selection.cross_val_score(clf, TEST, Y_test, cv=5,scoring='accuracy')
    print("Accuracy crossval: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    #f1-macro
    scores = model_selection.cross_val_score(clf, TEST, Y_test, cv=5,scoring='f1_macro')
    print("f1-macro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    #Obtaining predictions by cross-validation¶
    predicted = model_selection.cross_val_predict(clf, TEST, Y_test, cv=5)
    scores=metrics.accuracy_score(Y_test, predicted)
    print("Obtaining predictions by cross-validation: %0.2f" % scores)




    target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7',
                    'class 8', 'class 9', 'class 10', 'class 11', 'class 12', 'class 13', 'class 14',
                    'class 15', 'class 16', 'class 17', 'class 18', 'class 19', 'class 20', 'class 21',
                    'class 22', 'class 23']
    print("classification_report")
    print(metrics.classification_report(Y_test, Y_pred, target_names=target_names))



    #===================
    # CONFUSSION MATRIX
    #===================
    print('\nCONFUSSION MATRIX')
    cm=metrics.confusion_matrix(Y_test,Y_pred)
    #print(cm)
    plt.matshow(cm)
    plt.title('Matriz de confusión')
    plt.colorbar()
    plt.ylabel('Valor real')
    plt.xlabel('Valor predicho')
    plt.show()
