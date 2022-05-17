# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 01:24:36 2020

@author: esakarya
"""
import numpy as np

def convertToBags(currentData):
    bag_IDs = list(currentData.bagID.unique())
    my_labels = [currentData[currentData.bagID==eachID].response.max() for eachID in bag_IDs]
    
    for i in range(len(my_labels)):
        if my_labels[i]==1:
            continue
        elif my_labels[i]==-1:
            my_labels[i]=0
    labels=np.array([[i] for i in my_labels],dtype="uint8")

    bags = []
    for i in bag_IDs:
        npBag=currentData[currentData.bagID==i][currentData.columns[3:]].values
        bags.append(npBag)
   
    return bags,labels

