# -*- coding: utf-8 -*-
"""
Created on Tue May 17 12:31:52 2022

@author: esakarya
"""

from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd 
import time
from sys import path
from os import getcwd
path.append(getcwd() + "/src")
path.append(getcwd() + "/src/EMDD_SMIL")
from EMDD_SMIL.load_data import convertToBags
from EMDD_SMIL.EMDD import EMDD
from functions import prepareCVData

def EMDD_(cvInput):
    
    model = EMDD()
    trainingData,testData = cvInput[1],cvInput[2]
    
    X_train,Y_train = convertToBags(trainingData)
    X_test,Y_test  = convertToBags(testData)

    startTime=time.perf_counter()
    model.fit(X_train,Y_train)

    train_predictions = model.predict(X_train)
    predictions = model.predict(X_test)

    if (isinstance(predictions, tuple)):
        predictions = predictions[0]

    if (isinstance(train_predictions, tuple)):
        train_predictions = train_predictions[0]
    

    trainAUC=roc_auc_score(Y_train,train_predictions)
    testAUC =roc_auc_score(Y_test,predictions)

    #Calculation of Accuracy
    accuracy = np.average(Y_test.T == np.sign(predictions)) 
    runTime = time.perf_counter()-startTime
    return   trainAUC,testAUC,accuracy,runTime

def EMDD_cv(data_dir,dataFiles,replications,folds):
    colnames =["data","replication","fold","trainRuntime","trainAUC","testAUC","Accuracy"]   
    EMDD_results = pd.DataFrame(columns=colnames)
    for dataName in dataFiles:
        for rep in replications:
            wholeData = pd.read_excel(data_dir+'/'+dataName+'.xlsx')
            for f_ in folds:
                cvInput = prepareCVData(dataName, wholeData, rep, f_)
                try:
                    trainAUC,testAUC,accuracy,trainRuntime = EMDD_(cvInput)
                    currentSolutionValues=pd.DataFrame([[dataName,rep,f_,trainRuntime,trainAUC,testAUC,accuracy]], columns=EMDD_results.columns)
                    EMDD_results = EMDD_results.append(currentSolutionValues)
                except:
                    print("Error in ",dataName," replication",rep," fold",f_)
        averageOfdata=pd.DataFrame([[dataName,"Average","Average",EMDD_results[EMDD_results.data==dataName].trainRuntime.mean(),EMDD_results[EMDD_results.data==dataName].trainAUC.mean(),EMDD_results[EMDD_results.data==dataName].testAUC.mean(),EMDD_results[EMDD_results.data==dataName].Accuracy.mean()]], columns=EMDD_results.columns)
        EMDD_results = EMDD_results.append(averageOfdata)
        EMDD_results.to_csv("Results/EMDD_results.csv")