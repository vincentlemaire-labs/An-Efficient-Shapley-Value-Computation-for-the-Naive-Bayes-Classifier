
import os.path
import subprocess
import time
import pathlib
import pandas as pd
import numpy as np

# 10 datasets
DataName=['Twonorm','Crx','Ionosphere','Spam','Tictactoe','German','Telco','Adult','KagRiskFactorsCervicalCancer','Breast']
Target=['Class','Class','Class','spam','Class','Class','Churn','class','Biopsy','Class']
NbVar=[20,15,34,57,9,24,21,15,35,10]
NBClasses=[2,2,2,2,2,2,2,2,2,2,2,3,4,6,7,8,10]


# Two Class Datasets
# For all the datasets in the list or one in particular, here for Adult
for i in range(7,8):
    print(DataName[i])
    DATAWOE= '.\\experiments\\'+DataName[i]+'\\'+ 'Interpretation_WOE.txt'
    DATASHAP= '.\\experiments\\'+DataName[i]+'\\'+ 'Interpretation_SHAP.txt'
    MODELINGKDIC= '.\\experiments\\'+DataName[i]+'\\Modeling.kdic'
    FOLDER='.\\experiments\\'+DataName[i]
    rootFOLDER='.\\experiments\\'
    

    dfwoe = pd.read_csv(DATAWOE, sep="\t")
    dfshap = pd.read_csv(DATASHAP, sep="\t")
    
    print(dfshap.shape)

    shape = dfshap.shape
    nbcol=shape[1]
    dfshap2=pd.DataFrame()
    for j in range(nbcol):
        nom=dfshap.columns[j]
        result = nom.startswith('Shapley')
        if (result==True):
            dfshap2[nom] = dfshap[dfshap.columns[j]]

    shape = dfwoe.shape
    nbcol=shape[1]
    dfwoe2=pd.DataFrame()
    for j in range(nbcol):
        nom=dfwoe.columns[j]
        result = nom.startswith('WeightEvidence')
        if (result==True):
            dfwoe2[nom] = dfwoe[dfwoe.columns[j]]
            

    shape = dfwoe2.shape
    nbcol=shape[1]
    nbcol=int(nbcol)
    print('Nb Var in the ANB ')
    print(nbcol)
    
    # correlation per variable
    c=[0]*nbcol
    for j in range(nbcol):
        c[j]=dfshap2[dfshap2.columns[j]].corr(dfwoe2[dfwoe2.columns[j]])

    meanc=np.average(c)
    varc=np.var(c)
    
    # result per dataset
    namefile= FOLDER + '\Correlations.txt'
    f1 = open(namefile,"w")
    for j in range(nbcol):
        f1.write('Coeff' + str(j) + ': ' + '\t' + str(c[j]) +'\n')
    f1.write('\n')
    f1.write('Mean : ' + str(meanc) + '\n')
    f1.write('Variance : ' + str(varc) + '\n\n')
    
    # compute global WoE values
    dfwoe2abs=dfwoe2.abs()
    aa=dfwoe2abs.mean()
    shape = dfwoe2abs.shape
    nbcol=shape[1]
    for j in range(nbcol):
        f1.write('WoE Global' + str(j) + ': ' + '\t' + str(aa[j]) +'\n')

    # compute global shapley values
    dfshap2abs=dfshap2.abs()
    aa=dfshap2abs.mean()
    #print(aa)
    shape = dfshap2abs.shape
    #print(shape)
    nbcol=shape[1]
    for j in range(nbcol):
        f1.write('Shapley Global' + str(j) + ': ' + '\t' + str(aa[j]) +'\n')
        
    f1.close()

    
    # resume result for all dataset
    namefile= rootFOLDER + 'ALL_Correlations.txt'   
    f2 = open(namefile,"a")
    f2.write(DataName[i] +  '\t' +  str(nbcol/NBClasses[i]) +  '\t' + str(meanc) + '\t' + str(varc) + '\n')


    
    
f2.close()
