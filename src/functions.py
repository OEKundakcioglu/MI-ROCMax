# -*- coding: utf-8 -*-
"""
Created on Tue May 17 10:12:15 2022

@author: esakarya
"""

import time
# import pandas as pd
import numpy as np
from numpy import loadtxt
from sklearn.preprocessing import MinMaxScaler
scaling=True

def calcACC(trainingData,testData,weightVector):
    scoredData = trainingData.copy()
    scoredData['currentScores'] = np.dot(scoredData[scoredData.columns[3:]].values,weightVector)
    positives = np.array(list(scoredData[scoredData.response>=0.5].groupby("bagID").max().currentScores)) #2 Select witness instances for each positive bag (a.k.a highest scored positive). 
    negatives = np.array(list(scoredData[scoredData.response<=0.5].groupby("bagID").max().currentScores)) #3 Find highest scored negative instance in each bag(If positive witness instance is greater than the highest scored negative, then it is greater than all negatives in that bag.)
    totalInstances = np.append(positives,negatives)
    totalInstances.sort()
    
    accuracy = 0
    accuracyThreshold =totalInstances[0]
    for i in range(len(totalInstances)):
        currentAccuracy = (np.sum(negatives<totalInstances[i]) + np.sum(positives>=totalInstances[i])) / len(totalInstances)
        if currentAccuracy > accuracy:
            accuracy=currentAccuracy
            if(i+1 < len(totalInstances)-1):
                accuracyThreshold = (totalInstances[i]+totalInstances[i+1])/2
            else:
                accuracyThreshold = totalInstances[i]

    test_scoredData = testData.copy()
    test_scoredData['currentScores'] = np.dot(test_scoredData[test_scoredData.columns[3:]].values,weightVector)
    test_positives = np.array(list(test_scoredData[test_scoredData.response>=0.5].groupby("bagID").max().currentScores)) #2 Select witness instances for each positive bag (a.k.a highest scored positive). 
    test_negatives = np.array(list(test_scoredData[test_scoredData.response<=0.5].groupby("bagID").max().currentScores))
    
    test_accuracy = (np.sum(test_negatives<accuracyThreshold) + np.sum(test_positives>=accuracyThreshold))/ len(np.append(test_positives,test_negatives))
    
    return accuracy,test_accuracy

def calcAUC(data,weightVector):
    stAUCPC = time.perf_counter()
    
    scoredData = data.copy()
    scoredData['currentScores'] = np.dot(scoredData[scoredData.columns[3:]].values,weightVector) #1 Calculate score for each instance.
    witnesses = list(scoredData[scoredData.response>=0.5].groupby("bagID").max().currentScores) #2 Select witness instances for each positive bag (a.k.a highest scored positive). 
    highestNegatives = list(scoredData[scoredData.response<=0.5].groupby("bagID").max().currentScores) #3 Find highest scored negative instance in each bag(If positive witness instance is greater than the highest scored negative, then it is greater than all negatives in that bag.)
    AUC = sum([witness>highestNegative for witness in witnesses for highestNegative in highestNegatives])/(len(witnesses)*len(highestNegatives)) #4 Make pairwise comparison between all witness instances and all highest scored negative instances within bags.
    testTime = time.perf_counter()-stAUCPC
    return AUC,testTime

def calcObjective(data,weightVector,posBags,negBags):
    #1 Calculate scores for each instance.
    #2 Select witness instances for each positive bag (a.k.a highest scored positive). 
    #3 Find highest scored negative instance in each bag(If positive witness instance is greater than the highest scored negative, then it is greater than all negatives in that bag.)
    #4 Make pairwise comparison between all witness instances and all highest scored negative instances within bags.
    #positives and negatives are the id's of highest scored instances in each bags
    
    witnesses = []
    highestNegatives= []
    
    scoredData = data.copy()
    scoredData['currentScores'] = np.dot(scoredData[scoredData.columns[3:]].values,weightVector)
    
    for eachPosBag in posBags:
        greatestPosScoreInBag = np.array(scoredData[scoredData.bagID==eachPosBag].currentScores).max()
        witnesses.append(greatestPosScoreInBag)
        
    for eachNegBag in negBags:
        greatest_Neg_ScoreInBag = np.array(scoredData[scoredData.bagID==eachNegBag].currentScores).max()
        highestNegatives.append(greatest_Neg_ScoreInBag)
    pairComp = sum([witness>highestNegative for witness in witnesses for highestNegative in highestNegatives])
    return pairComp

def prepareCVData(dataName, wholeData, rep, f_):
    #Read the testing instances
    testBagIDs= loadtxt('DATASETS/'+dataName+'.csv_rep'+rep+'_fold'+f_+'.txt',unpack=False)
    #Test data
    testData = wholeData.loc[wholeData.bagID.isin(testBagIDs)].copy()
    test_instances=testData.instance.tolist()
    #Training data
    trainingData=wholeData.drop(test_instances).copy()
    trainingData.reset_index(inplace = True,drop=True)
    trainingData.instance = list(range(len(trainingData)))
    
    if scaling:
        scaler = MinMaxScaler()
        trainingData[trainingData.columns[3:]] = scaler.fit_transform(trainingData[trainingData.columns[3:]])
        testData[testData.columns[3:]] = scaler.transform(testData[testData.columns[3:]])
    
    trainPoints = trainingData.values  # point coordinates
    trainNumInstances = len(trainPoints)
    
    negInstances=list(set(trainingData[trainingData.response<1].instance))
    posInstances=list(set(trainingData[trainingData.response>0].instance))
    
    negBagSet=list(set(trainingData[trainingData.response<1].bagID)) #ID's negative bags
    posBagSet=list(set(trainingData[trainingData.response>0].bagID)) #ID's positive bags
    
    trainBagDict={i:list(trainingData.instance[trainingData.bagID==i]) for i in list(set(trainingData.bagID))}

    #First three columns of data: instance-response-bagID
    train_Cord_ij = trainingData.values[:,3:] #Feature matrix of training data
    numVector = len(train_Cord_ij[0,:]) #Number of features

    return numVector,trainingData,testData, posInstances,negInstances,posBagSet,negBagSet,trainBagDict,trainNumInstances,train_Cord_ij

def fixVals(val):
    if val<1:
        return -1
    else:
        return 1