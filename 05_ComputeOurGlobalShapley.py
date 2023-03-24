
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
    DATASHAP= '.\\experiments\\'+DataName[i]+'\\'+ 'Interpretation_SHAP.txt'
    FOLDER='.\\experiments\\'+DataName[i]
    rootFOLDER='.\\experiments\\'
    KERNELSHAPVALUES= '.\\experiments\\'+DataName[i]+'\\'+'OurGlobalShap_'+DataName[i]+'.txt'

    dfshap = pd.read_csv(DATASHAP, sep="\t")
    
    
    print("Fin chargement fichier")
    # load file and  keep only columns of variable importances
    shape = dfshap.shape
    nbcol=shape[1]
    dfshap2=pd.DataFrame()
    for j in range(nbcol):
        nom=dfshap.columns[j]
        result = nom.startswith('Shapley')
        if (result==True):
            dfshap2[nom] = dfshap[dfshap.columns[j]]
            
    

    #save the gloabl shap values in a file 
    namefile= KERNELSHAPVALUES
    f1 = open(namefile,"w")
    shape = dfshap2.shape
    nbcol=int(shape[1]/NBClasses[i])
    f1.write('Nb Classes : ' + str(NBClasses[i]) + '\n')
    f1.write('Nb Vars : ' + str(nbcol) + '\n')
    vals = np.abs(dfshap2.values).mean(0)
    
    for k in range(0,NBClasses[i]):
        for l in range(0,nbcol):
            #print(str(k*nbcol+l))
            f1.write(str(vals[k*nbcol+l]) + '\t')
        f1.write('\n')
         
    f1.close()
