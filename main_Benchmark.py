# -*- coding: utf-8 -*-
"""
Created on Tue May 17 10:15:51 2022

@author: esakarya
"""
# from __future__ import print_function, division
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
from src.MI_SVM import MISVM_cv
from src.miSVM import miSVM_cv
# from src.cKNN import cKNN_cv
# from src.KNN_ import KNN_cv
# from src.EMDD_ import EMDD_cv
# from src.simpleMIL_ import simpleMIL_cv
# from src.MI_NET import MI_NET_cv
# from src.MI_NET_RC import MI_NET_RC_cv
# from src.MI_NET_Attention import MI_NET_Attention_cv
# from src.miNET import miNET_cv

dataFiles = ["Musk1","Musk2","Tiger","Fox","Elephant","CorelAntique","CorelBattleships","CorelBeach","Mutagenesis1","Mutagenesis2","Newsgroups1","UCSBBreastCancer"]
replications=['1','2','3','4','5']
folds=['1','2','3','4','5','6','7','8','9','10']

# MISVM_cv(data_dir='DATASETS',dataFiles=dataFiles,replications=replications,folds=folds)

# miSVM_cv(data_dir='DATASETS',dataFiles=dataFiles,replications=replications,folds=folds)

# cKNN_cv(data_dir='DATASETS',dataFiles=dataFiles,replications=replications,folds=folds)

# KNN_cv(data_dir='DATASETS',dataFiles=dataFiles,replications=replications,folds=folds)

# EMDD_cv(data_dir='DATASETS',dataFiles=dataFiles,replications=replications,folds=folds)

# simpleMIL_cv(data_dir='DATASETS',dataFiles=dataFiles,replications=replications,folds=folds)

dataFiles = ["CorelAntique","CorelBattleships","CorelBeach","Mutagenesis1","Mutagenesis2"]
# MI_NET_cv(data_dir_= 'DATASETS',Datasets=dataFiles,replications=replications,folds=folds)

# MI_NET_RC_cv(data_dir_= 'DATASETS',Datasets=dataFiles,replications=replications,folds=folds)

# MI_NET_Attention_cv(data_dir_= 'DATASETS',Datasets=dataFiles,replications=replications,folds=folds)

# miNET_cv(data_dir_= 'DATASETS',Datasets=dataFiles,replications=replications,folds=folds)

# MISVM_cv(data_dir='DATASETS',dataFiles=dataFiles,replications=replications,folds=folds)

# miSVM_cv(data_dir='DATASETS',dataFiles=dataFiles,replications=replications,folds=folds)

