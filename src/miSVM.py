# -*- coding: utf-8 -*-
"""
Created on Tue May 17 10:08:08 2022

@author: esakarya
"""
from __future__ import print_function, division
import numpy as np
import misvm
import time
import pandas as pd 
from numpy import loadtxt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
import warnings
from sklearn.exceptions import DataConversionWarning
from src.functions import prepareCVData,fixVals
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

def miSVM(cvInput):
    start_time=time.perf_counter()
    
    trainingData,testData = cvInput[1],cvInput[2]
    classifier= misvm.miSVM(kernel='linear', C=20.0, max_iters=20)

    trainBags,trainLabels,testBags,testLabels = [],[],[],[]
    for bag in list(trainingData.bagID.unique()):
        trainBags.append(trainingData.loc[trainingData.bagID==bag][trainingData.columns[3:]].values)
        trainLabels.append(max(list(trainingData.loc[trainingData.bagID==bag].response)))
    trainLabels = np.array(trainLabels)
    for bag in list(testData.bagID.unique()):
        testBags.append(testData.loc[testData.bagID==bag][testData.columns[3:]].values)
        testLabels.append(list(testData.loc[testData.bagID==bag].response)[0])
    testLabels= np.array(testLabels)
    
    classifier.fit(trainBags, trainLabels)
    train_predictions = classifier.predict(trainBags)
    predictions = classifier.predict(testBags)
    accuracy=np.average(testLabels == np.sign(predictions))

    trainAUC=roc_auc_score(trainLabels,train_predictions)
    testAUC =roc_auc_score(testLabels,predictions)
    
    runTime = time.perf_counter()-start_time
    
    return trainAUC,testAUC,accuracy,runTime

def miSVM_cv(data_dir,dataFiles,replications,folds):
    colnames =["data","replication","fold","trainRuntime","trainAUC","testAUC","Accuracy"]   
    miSVM_results = pd.DataFrame(columns=colnames)
    for dataName in dataFiles:
        for rep in replications:
            wholeData = pd.read_excel(data_dir +'/'+dataName+'.xlsx')
            wholeData.response = np.array([fixVals(label) for label in wholeData.response])
            for f_ in folds:
                cvInput = prepareCVData(dataName, wholeData, rep, f_)
                try:
                    start_time=time.perf_counter()
                    
                    trainingData,testData = cvInput[1],cvInput[2]
                    classifier = misvm.miSVM(kernel='linear',  C=20.0, max_iters=20)
            
                    trainBags,trainLabels,testBags,testLabels = [],[],[],[]
                    for bag in list(trainingData.bagID.unique()):
                        trainBags.append(trainingData.loc[trainingData.bagID==bag][trainingData.columns[3:]].values)
                        trainLabels.append(max(list(trainingData.loc[trainingData.bagID==bag].response)))
                    trainLabels = np.array(trainLabels)
                    for bag in list(testData.bagID.unique()):
                        testBags.append(testData.loc[testData.bagID==bag][testData.columns[3:]].values)
                        testLabels.append(list(testData.loc[testData.bagID==bag].response)[0])
                    testLabels= np.array(testLabels)
                    
                    classifier.fit(trainBags, trainLabels)
                    train_predictions = classifier.predict(trainBags)
                    predictions = classifier.predict(testBags)
                    accuracy=np.average(testLabels == np.sign(predictions))

                    trainAUC=roc_auc_score(trainLabels,train_predictions)
                    testAUC =roc_auc_score(testLabels,predictions)
                    
                    trainRuntime = time.perf_counter()-start_time
                    currentSolutionValues=pd.DataFrame([[dataName,rep,f_,trainRuntime,trainAUC,testAUC,accuracy]], columns=miSVM_results.columns)
                    miSVM_results = miSVM_results.append(currentSolutionValues)
                   
                except:
                    print("\n\n\n Error in ",dataName," replication",rep," fold",f_)
        averageOfdata=pd.DataFrame([[dataName,"Average","Average",miSVM_results[miSVM_results.data==dataName].trainRuntime.mean(),miSVM_results[miSVM_results.data==dataName].trainAUC.mean(),miSVM_results[miSVM_results.data==dataName].testAUC.mean(),miSVM_results[miSVM_results.data==dataName].Accuracy.mean()]], columns=miSVM_results.columns)
        miSVM_results = miSVM_results.append(averageOfdata)
        miSVM_results.to_csv("Results/miSVM_Results.csv")