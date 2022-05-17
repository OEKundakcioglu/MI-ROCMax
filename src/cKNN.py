# -*- coding: utf-8 -*-
"""
Created on Tue May 17 11:54:36 2022

@author: esakarya
"""
import time
from sys import path
from os import getcwd
path.append(getcwd() + "/src")
from Citation_KNN.load_data import convertToBags
from Citation_KNN.CitationKNN import CitationKNN
from sklearn.metrics import roc_auc_score
from functions import prepareCVData
import numpy as np
import pandas as pd

def cKNN(cvInput,parameters):
    startTime=time.perf_counter()

    classifier = CitationKNN() 
    trainingData,testData = cvInput[1],cvInput[2]
    
    X_train,Y_train = convertToBags(trainingData)
    X_test,Y_test  = convertToBags(testData)
    
    if len(parameters) > 0: 
        classifier.fit(X_train, Y_train, **parameters)
    else:
        classifier.fit(X_train, Y_train)
    
    train_predictions = classifier.predict(X_train)
    predictions = classifier.predict(X_test)

    if (isinstance(predictions, tuple)):
        predictions = predictions[1]
    
    if (isinstance(train_predictions, tuple)):
        train_predictions = train_predictions[1]
    
    runTime = time.perf_counter()-startTime
    
    trainAUC=roc_auc_score(Y_train,train_predictions)
    testAUC =roc_auc_score(Y_test,predictions)
    accuracy = np.average(Y_test.T == np.sign(predictions))
    
    return trainAUC,testAUC,accuracy,runTime

def cKNN_cv(data_dir,dataFiles,replications,folds,parameters={'references': 2, 'citers': 4}):
    # parameters={'references': k, 'citers': k+2}
    colnames =["data","replication","fold","trainRuntime","trainAUC","testAUC","Accuracy"]   
    cKNN_results = pd.DataFrame(columns=colnames)
    for dataName in dataFiles:
        for rep in replications:
            wholeData = pd.read_excel(data_dir+'/'+dataName+'.xlsx')
            for f_ in folds:
                cvInput = prepareCVData(dataName, wholeData, rep, f_)
                try:
                    trainAUC,testAUC,accuracy,trainRuntime = cKNN(cvInput,parameters)
                    currentSolutionValues=pd.DataFrame([[dataName,rep,f_,trainRuntime,trainAUC,testAUC,accuracy]], columns=cKNN_results.columns)
                    cKNN_results = cKNN_results.append(currentSolutionValues)
                except:
                    print("Error in ",dataName," replication",rep," fold",f_)
        averageOfdata=pd.DataFrame([[dataName,"Average","Average",cKNN_results[cKNN_results.data==dataName].trainRuntime.mean(),cKNN_results[cKNN_results.data==dataName].trainAUC.mean(),cKNN_results[cKNN_results.data==dataName].testAUC.mean(),cKNN_results[cKNN_results.data==dataName].Accuracy.mean()]], columns=cKNN_results.columns)
        cKNN_results = cKNN_results.append(averageOfdata)
        cKNN_results.to_csv("Results/cKNN_Results.csv")