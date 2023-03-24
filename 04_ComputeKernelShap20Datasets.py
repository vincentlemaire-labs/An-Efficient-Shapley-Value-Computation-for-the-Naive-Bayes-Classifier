
import os
import pickle
from os import path

import pandas as pd
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
# shap has to be installed before
import shap
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import timeit

# khiops should be install before
# and also pykhiops using the pip install see the file readme.txt
from pykhiops import core as pk
from pykhiops.sklearn import (
    KhiopsClassifier
)

# 10 datasets
DataName=['Twonorm','Crx','Ionosphere','Spam','Tictactoe','German','Telco','Adult','KagRiskFactorsCervicalCancer','Breast']
Target=['Class','Class','Class','spam','Class','Class','Churn','class','Biopsy','Class']
NbVar=[20,15,34,57,9,24,21,15,35,10]
NBClasses=[2,2,2,2,2,2,2,2,2,2,2,3,4,6,7,8,10]

# Two Class Datasets
# For all the datasets in the list or one in particular, here for Adult
for i in range(7,8):

    FOLDER='.\\experiments\\'+DataName[i]
    DATA= '.\\experiments\\'+DataName[i]+'\\'+DataName[i]+'.txt'
    KERNELSHAPVALUES= '.\\experiments\\'+DataName[i]+'\\'+'KernelShap_'+DataName[i]+'.txt'

    print(DATA)
    
    data_df = pd.read_csv(DATA, sep="\t")

    #define the data and the target
    X_train = data_df.drop([Target[i]], axis=1)      
    Y_train = data_df[Target[i]]
    
    #define the ANB classifier using Khiops (a single Naive Bayes with no tree , see the khiops documentation)
    pkc = KhiopsClassifier(n_trees=0)

    #fit the classifier
    # it will be save in temporary folder
    pkc.fit(X_train, Y_train)  
    
    #keep only variables used by the classifier
    X_train2=pd.DataFrame()
    main_dico = pkc.model_.get_dictionary(pkc.model_main_dictionary_name_)
    for var in main_dico.variables :
        if "Weight" in var.meta_data:
            X_train2[var.name]=X_train[var.name]
    X_train=X_train2
    
    # fit again the classifier to be trained only with informative variables
    # it will be save in temporary folder
    # this is to not after use kernelshap with unused variables
    pkc.fit(X_train, Y_train)  
    
    #init shap
    shap.initjs()
    
    #Khiops is only define for dataframe but kernelshap uses numpy
    # so we need to convert numpy to dataframe when using the predict function of khiops
    def pkc2(x):
        #first ste convert the numpy to dataframe with the right name columns
        x2=pd.DataFrame(x,columns=X_train.columns)
        #second step set the right type to be compliant with the fit of the classifier
        x2=x2.astype(X_train.dtypes.to_dict())
        y=pkc.predict_proba(x2)
        return y

    # sampling of 1000 examples
    print('N= ',X_train.shape[0])
    if (X_train.shape[0]>1000):
        # draw 1000 examples with a given seed 
        X_train=X_train.sample(1000,random_state=1)
    print('Sampling of 1000 examples if N greater than 1000')
    print('N= ',X_train.shape[0])
    
    # we do a loop to find how many examples we may keep in the
    # knowledge data table to have the complete result in 2 hours
    #number of examples to explaine
    nb_examples=X_train.shape[0]
    # 2 hours max 
    nbsecondeperexample=7200/nb_examples
    print('nb secondes allowed :' , nbsecondeperexample)
    nn=0
    elapsed=0
    while True:
        nn=nn+50
        shap.initjs()
        data=shap.sample(X_train, nn)
        pkc_explainer = shap.KernelExplainer(pkc2,data)
        start_time = timeit.default_timer()
        pkc_shap_values = pkc_explainer.shap_values(X_train.iloc[0])
        elapsed = timeit.default_timer() - start_time
        
        if (elapsed > nbsecondeperexample or nn>nb_examples):
            if (nn>nb_examples):
                nn=nb_examples
            break    
    print('final nn ',nn)

    # now we know the value we set the explainer
    data=shap.sample(X_train, nn)
    pkc_explainer = shap.KernelExplainer(pkc2,data)

    # compute the shapley values
    start_time = timeit.default_timer()
    pkc_shap_values = pkc_explainer.shap_values(X_train)
    elapsed = timeit.default_timer() - start_time

    #number of classes
    c1=len(pkc_shap_values)
    
    #save the shap values in a file 
    namefile= KERNELSHAPVALUES
    f1 = open(namefile,"w")
    feature_names = X_train.columns
    f1.write('Nb examples data-kowledge=' + str(nn) + '\n')
    f1.write('Time used to compute' + str(elapsed) + '\n')
    for z in range(0,len(feature_names)):
            f1.write(feature_names[z]+'\t')
    f1.write('\n')
    for k in range(0,c1):
        shap_df = pd.DataFrame(pkc_shap_values[k], columns = feature_names)
        vals = np.abs(shap_df.values).mean(0)
        for z in range(0,len(vals)):
            f1.write(str(vals[z])+'\t')
        f1.write('\n')
    f1.close()

