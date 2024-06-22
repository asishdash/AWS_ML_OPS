import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import logging
from urllib.parse import urlparse


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)



def eval_metrics(actual,pred):
    accuracy=metrics.accuracy_score(actual,pred)
    print("here1")
    precision = metrics.precision_score(actual,pred, pos_label='M')
    print("here")
    recall = metrics.recall_score(actual,pred ,pos_label='M')
    print("here2")
    return accuracy,precision,recall


def logRegModel():
    model = LogisticRegression()
    return model

def dtModel():
    model = DecisionTreeClassifier()
    return model

def rfmodel():
    params={'max_depth':6,'random_state':1010}
    model = RandomForestClassifier()
    return model,params

def create_split_feature(pred_var):
    train_X =  train[pred_var]
    train_Y =  train.diagnosis
    test_X  =  test[pred_var]
    test_Y  =  test.diagnosis
    return train_X,train_Y,test_X,test_Y 



if __name__ =="__main__":
    #Warning.filter ('ignore')
    np.random.seed(1010)
    data = pd.read_csv('/Users/asishdash/Documents/AD/learn/git_repo/AWS_ML_OPS/Breast_Cancer_Analysis/data.csv')
    train,test = train_test_split(data,test_size=0.3,random_state=1983)
    data.drop('id',axis=1,inplace=True)
    data.drop('Unnamed: 32',axis=1,inplace=True)
    data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})
    pred_var_mean =[x for x in data.columns if  "_mean"in x ]
    pred_var_se =[x for x in data.columns if  "_se"in x ]
    pred_var_worst =[x for x in data.columns if  "_worst"in x ]
    train_X,train_Y,test_X,test_Y  = create_split_feature(pred_var_se)

    with mlflow.start_run():
        #model=logRegModel()
        #model = DecisionTreeClassifier()
        model,params = rfmodel()
        model.fit(train_X,train_Y)
        predictions = model.predict(test_X)
        (accuracy,precision,recall) = eval_metrics(test_Y,predictions)
        mlflow.log_metric("ACC",accuracy)
        mlflow.log_metric("Precission",precision)
        mlflow.log_metric("Recall",recall)
        mlflow.log_params(params)
        predictions=model.predict(train_X)
        signature = infer_signature(train_X,predictions)
        tracking_url_type_store =urlparse(mlflow.get_tracking_uri() ).scheme
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="Mean Run",
            #registered_model_name="Logistic Regression",
            registered_model_name="Random Forest",
            signature= signature 
        )

