from sklearn.metrics import roc_auc_score
from mil.utils import get_raw_train_test_data
from mil.dataloader import get_train_test_dataloader
from mil.nn.models import mi_Net
from mil.train import train, eval_model
from src.functions import prepareCVData
import pandas as pd
import itertools
import time


def miNET_cv(data_dir_,Datasets,replications,folds):
    # replications=['1','2','3','4','5']
    # folds=['1','2','3','4','5','6','7','8','9','10']
    rep_fold = list(itertools.product(replications,folds))
    colnames = ["data","replication","fold","trainRuntime","trainAUC","testAUC","trainAccuracy","TestAccuracy"]  
    results = pd.DataFrame(columns=colnames)
    for dataSet in Datasets:
        if dataSet=="Musk1":
            learning_rate,weight_decay,momentum=5e-4,5e-3,0.9
        elif dataSet=="Musk2":
            learning_rate,weight_decay,momentum=5e-4,3e-2,0.9
        elif dataSet=="Fox":
            learning_rate,weight_decay,momentum=1e-4,1e-2,0.9
        elif dataSet=="Tiger":
            learning_rate,weight_decay,momentum=5e-4,5e-2,0.9
        elif dataSet=="Elephant":
            learning_rate,weight_decay,momentum=1e-4,5e-2,0.9
        else:
            learning_rate,weight_decay,momentum=0.00034,0.011,0.9
        for rep,fold in rep_fold:
            startTime = time.perf_counter()
            batch_size = 1
            data_dir ='DATASETS/'+dataSet+'.xlsx'
            test_index_dir = 'DATASETS/'+dataSet+".csv_rep"+rep+"_fold"+fold+".txt"
            inputDim = prepareCVData(dataSet, pd.read_csv('DATASETS/'+dataSet+'.csv'), rep, fold)[0]
            train_data, test_data = get_raw_train_test_data(data_dir=data_dir, test_index_dir=test_index_dir)
            train_dl, test_dl = get_train_test_dataloader(train_data=train_data, test_data=test_data, train_batch_size=batch_size, 
                test_batch_size=batch_size)
        
            model = mi_Net(input_dim=inputDim)   # build the model
            model = train(model=model, train_dl=train_dl,learning_rate=learning_rate,weight_decay=weight_decay,momentum=momentum)
            train_acc, train_pred_prob,trainLabels = eval_model(model=model, dataloader=train_dl)
            test_acc, test_pred_prob,testLabels = eval_model(model=model, dataloader=test_dl)
            
        
            trainAUC = roc_auc_score(trainLabels,train_pred_prob)
            testAUC = roc_auc_score(testLabels,test_pred_prob)
            totalTime = time.perf_counter() - startTime
            iterVals=pd.DataFrame([[dataSet,rep,fold, totalTime,trainAUC,testAUC,train_acc,test_acc]], columns=results.columns)
            results = results.append(iterVals)
            results.to_csv("Results/mi__NET.csv")
        averageOfdata=pd.DataFrame([[dataSet,"Average","Average",results[results.data==dataSet].trainRuntime.mean(),results[results.data==dataSet].trainAUC.mean(),results[results.data==dataSet].testAUC.mean(),results[results.data==dataSet].trainAccuracy.mean(),results[results.data==dataSet].TestAccuracy.mean()]], columns=results.columns)
        results = results.append(averageOfdata)
        results.to_csv("Results/mi__NET.csv")
    results.to_csv("Results/mi__NET.csv")

