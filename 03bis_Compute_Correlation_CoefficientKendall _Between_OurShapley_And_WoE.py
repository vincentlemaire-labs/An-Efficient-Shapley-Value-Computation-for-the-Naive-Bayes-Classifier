

import os.path
import subprocess
import time
import pathlib
import pandas as pd
import numpy as np
import scipy.stats as stats

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
    # load file 'SHAP' computed in step 2 and keep only variable inportances
    shape = dfshap.shape
    nbcol=shape[1]
    dfshap2=pd.DataFrame()
    for j in range(nbcol):
        nom=dfshap.columns[j]
        result = nom.startswith('Shapley')
        if (result==True):
            dfshap2[nom] = dfshap[dfshap.columns[j]]
            
    # load file 'WoE' computed in step 2 and keep only variable inportances
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
    nbline=int(shape[0])
        
    #transpose the two dataframe to compute the kendall coefficent on lines
    dfwoe3=dfwoe2.transpose()
    dfshap3=dfshap2.transpose()
    shape = dfwoe3.shape
    nbcol=shape[1]
    
    # compute correlations line per line 
    c=[0]*nbcol
    p=[0]*nbcol
    for j in range(nbcol):
        x1=dfshap3[dfshap3.columns[j]]
        x2=dfwoe3[dfwoe3.columns[j]]
        tau, p_value = stats.kendalltau(x1, x2)
        c[j]=tau
        p[j]=p_value

    meanc=np.average(c)
    meanp=np.average(p)
    varc=np.var(c)
    
    # save results 
    namefile= FOLDER + '\CorrelationsKendall.txt'
    f1 = open(namefile,"w")
    for j in range(nbcol):
        f1.write('Coeff' + str(j) + ': ' + '\t' + str(c[j]) +'\n')
    f1.write('\n')
    f1.write('Mean coeff : ' + str(meanc) + '\n')
    f1.write('Variance : ' + str(varc) + '\n')
    f1.write('Mean p-value : ' + str(meanp) + '\n\n')
    
    #  mean absolute value for WOE
    dfwoe2abs=dfwoe2.abs()
    aa=dfwoe2abs.mean()
    shape = dfwoe2abs.shape
    nbcol=shape[1]
    for j in range(nbcol):
        f1.write('WoE Global' + str(j) + ': ' + '\t' + str(aa[j]) +'\n')

    # mean absolute value for Shapley
    dfshap2abs=dfshap2.abs()
    aa=dfshap2abs.mean()
    shape = dfshap2abs.shape
    nbcol=shape[1]
    for j in range(nbcol):
        f1.write('Shapley Global' + str(j) + ': ' + '\t' + str(aa[j]) +'\n')
        
    f1.close()

    
    # result resume for all dataset
    namefile= rootFOLDER + 'ALL_Correlations_kendall.txt'   
    f2 = open(namefile,"a")
    f2.write(DataName[i] +  '\t' +  str(nbcol/NBClasses[i]) +  '\t' + str(meanc) + '\t' + str(varc) +  '\t' + str(meanp) + '\n')


    
    
f2.close()
