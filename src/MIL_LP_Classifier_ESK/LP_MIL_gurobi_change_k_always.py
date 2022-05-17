
#!/usr/bin/python
# -*- coding: utf-8 -*-


# =============================================================================
# This code is taken from Kucukasci's website. We just fixed the bugs
# http://ww3.ticaret.edu.tr/eskucukasci/multiple-instance-learning/
# =============================================================================

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from gurobipy import Model,LinExpr,GRB,itertools,gc

import os
from os import listdir
import time
from scipy import sparse
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
scaling =True
cwd = os.getcwd()
datafolder=cwd+'/data/'
cvFolder=cwd+'/CV/'
resultsFolder=cwd+'/results/'
fil = listdir(datafolder)
fil.sort()
nfold=10
nrep=5

## select data representation alternative (either 'R^instance' or 'R^cluster'):

##dataRepresentation='R^instance'
dataRepresentation='R^cluster'

A = np.arange(0,len(fil))
colnames=['Dataset Name', 'Replication', 'Fold', 'TrainAUC', 'TestAUC','Accuracy', 'RepTime','ModelTime','SolutionTime','IterCount','ObjVal']
results=pd.DataFrame(columns = colnames)
cnt=0

for ind in A:
    print (fil[ind])
    # data initialization
    alldat=pd.read_csv(datafolder + fil[ind],header=None)   
    bagids=alldat[1].unique()
    nofinstance=alldat[1].value_counts(sort=False).values
    labels=alldat.groupby([1])[0].first().values
    labels[labels==0]=-1
    
    for rep in range(0, 5):
         for k in range(0, nfold):
            testind=np.loadtxt(cvFolder + fil[ind] + '_rep' + str(rep+1) + '_fold' + str(k+1) + '.txt')
            testind=testind.astype(int)
            noftestinstance=nofinstance[testind-1]
            testlabels=labels[testind-1]
            testcv=alldat[alldat.iloc[:,1].isin(testind)]
            traincv=alldat[-alldat.iloc[:,1].isin(testind)]
            trainind=traincv[1].unique()
            trainlabels=labels[trainind-1]
            noftraininstance=nofinstance[trainind-1]
            nofalltraininstance=traincv.shape[0]
            noftrainbags=noftraininstance.shape[0]
            nofalltestinstance=testcv.shape[0]

            traininstances=traincv.values[:,2:traincv.shape[1]]
            testinstances=testcv.values[:,2:testcv.shape[1]]
            
            if scaling:
                scaler=MinMaxScaler()
                traininstances=scaler.fit_transform(traininstances)
                testinstances = scaler.transform(testinstances)

            
            t = time.perf_counter()

            if dataRepresentation=='R^instance': #dissimilarity to the instances (R^instance)
                Xtrain = euclidean_distances(traininstances,traininstances)
                Xtest = euclidean_distances(testinstances,traininstances)

            if dataRepresentation=='R^cluster': #dissimilarity to the cluster centers (R^cluster)
                if (True):
                    clusterRange=np.arange(25, traininstances.shape[0], 25)
                    iter=0
                    improvement=10000
                    distortions = []
                    improvements = []
                    TermCriterion=0.01
                    SSE = []
                    while improvement>TermCriterion:
                        KM = KMeans(n_clusters=clusterRange[iter],random_state=0,init='k-means++',n_init=5, max_iter=200).fit(traininstances)
                        withinSSE = KM.inertia_
                        SSE.append(withinSSE)
                        if iter==0:
                            currentSSE=withinSSE
                            improvements.append(0)
                        else:
                            improvement=(currentSSE-withinSSE)/currentSSE
                            improvements.append(improvement)
                            currentSSE=withinSSE
                            SSE[iter] = currentSSE
                            if (iter>2) & (abs(improvements[iter-1]-improvements[iter])<TermCriterion) & (abs(improvements[iter-1]-improvements[iter-2])<TermCriterion):
                                break
                        iter=iter+1
                        if iter >= len(clusterRange)-1:
                            break
                    nofClust=clusterRange[iter]
                print(nofClust)

                kmeans = KMeans(n_clusters=nofClust, random_state=0,init='k-means++',n_init=5, max_iter=200).fit(traininstances)
                clustercenters = kmeans.cluster_centers_
                Xtrain = euclidean_distances(traininstances,clustercenters)
                Xtest = euclidean_distances(testinstances,clustercenters)
            
            noffeature=Xtrain.shape[1]
            rep_time=time.perf_counter()-t
           
            
            t = time.perf_counter()
            m = Model('MIL_LP')  # initiating model
            m.reset()
            m.update()

            w = m.addVars(noffeature,lb=-GRB.INFINITY, ub=GRB.INFINITY,name="w")
            c = m.addVar(name="c",lb=-GRB.INFINITY, ub=GRB.INFINITY)
            m_ins = m.addVars(nofalltraininstance,lb=0, ub=1,name="m_ins")
            m_bag = m.addVars(len(trainind),lb=-GRB.INFINITY, ub=GRB.INFINITY,name="m_bag")
            sigma=m.addVars(len(np.where(trainlabels==1)[0]),len(np.where(trainlabels==-1)[0]),lb=-GRB.INFINITY,ub=GRB.INFINITY,name="S")

            m.update()

            coefvars=np.append(w.select(),c) 
            X_cons=np.column_stack((Xtrain, np.ones(nofalltraininstance),-np.ones(nofalltraininstance)))
            
            for i in range(0, nofalltraininstance):
                m.addConstr(LinExpr(X_cons[i,:],np.append(coefvars,m_ins[i])), GRB.EQUAL,0)
                
            for i in range(0, len(trainind)):
                instanceInd=np.where(trainind[i]==traincv[1])
                nofbaginstance=len(instanceInd[0])
                m.addConstr(LinExpr(np.append(np.repeat(1, nofbaginstance),-nofbaginstance),np.append(m_ins.select(instanceInd),m_bag[i])),GRB.EQUAL,0)

            posbagIndex=np.where(trainlabels==1)[0]
            negbagIndex=np.where(trainlabels==-1)[0]
                
            m.addConstrs(-sigma[i,j] + m_bag[posbagIndex[i]] - m_bag[negbagIndex[j]] == 0 for i, j in itertools.product(range(len(posbagIndex)), range(len(negbagIndex))))
                        
            m.setObjective(LinExpr(sigma.sum()), GRB.MAXIMIZE)
            model_time=time.perf_counter()-t
            m.setParam('OutputFlag', False )
            m.setParam('Method', 2)
            m.setParam('BarConvTol', 0.01)
            m.setParam('Crossover', 0)
            m.update()
            
            t = time.perf_counter()
            m.optimize()
            solution_time=time.perf_counter()-t
            m.printQuality()
            weights=[w[i].X for i in range(0, noffeature)]
            cons=c.X
            testinstanceMemberships=np.dot(Xtest,weights)+cons
           
            newId=testcv[1].factorize()
            Mask = sparse.csr_matrix((np.ones(testcv.shape[0],int),(newId[0], np.arange(testcv.shape[0]))), shape=(testind.size,testcv.shape[0])).toarray()
            Mask=normalize(Mask, norm='l1', axis=1)
            testBagMemberships=np.dot(Mask,testinstanceMemberships)
            
            trainBagMemberships=[m_bag[i].X for i in range(0, len(trainind))]           
            trainroc=roc_auc_score(trainlabels,trainBagMemberships)
            testroc=roc_auc_score(testlabels,testBagMemberships)

            clf = RandomForestClassifier(n_estimators=1, random_state=0,max_depth =1)
            trBagMemb=np.array(trainBagMemberships)
            trBagMemb = trBagMemb[:, None]
            clf.fit(trBagMemb, trainlabels)
            tstBagMemb=np.array(testBagMemberships)
            tstBagMemb = tstBagMemb[:, None]
            accuracy=clf.score(tstBagMemb, testlabels)
            
            currentNofIter = m.IterCount
            objval = m.ObjVal
            # results.loc[cnt,:]= [fil[ind], rep+1, k+1, trainroc, testroc, accuracy, rep_time, model_time,solution_time, currentNofIter, objval]
            currentSolutionValues=pd.DataFrame([[fil[ind], rep+1, k+1, trainroc, testroc, accuracy, rep_time, model_time,solution_time, currentNofIter, objval]], columns=results.columns)
            results = results.append(currentSolutionValues)
            # print(results.loc[cnt])
            
            del m
            gc.collect()
            cnt+=1
    currentSolutionValues=pd.DataFrame([["Average","Average","Average",results.TrainAUC.mean(),results.TestAUC.mean(),results.Accuracy.mean(),results.RepTime.mean(),results.ModelTime.mean(),results.SolutionTime.mean(),results.IterCount.mean(),results.ObjVal.mean()]], columns=results.columns)
    results = results.append(currentSolutionValues)
    results.to_csv(resultsFolder+'LP_'+ dataRepresentation +'_change_k_always_' +str(fil[ind]))