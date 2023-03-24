
import subprocess
import time
import pathlib
import pickle
import os
import shutil

# khiops should be install before
from pykhiops import core as pk

# replace the MODL.exe with the on provided in this Github
# Providing two folder paths
origin = '.\MODL.exe'
# to know where the bin directory of khiops is :
tmp = pk.get_runner().khiops_bin_dir 
# target path
target = tmp + '\MODL.exe'
# replace the MODL.exe with the on provided in this Github
shutil.copy(origin, target)

# 10 datasets
DataName=['Twonorm','Crx','Ionosphere','Spam','Tictactoe','German','Telco','Adult','KagRiskFactorsCervicalCancer','Breast']
Target=['Class','Class','Class','spam','Class','Class','Churn','class','Biopsy','Class']
NbVar=[20,15,34,57,9,24,21,15,35,10]
NBClasses=[2,2,2,2,2,2,2,2,2,2,2,3,4,6,7,8,10]


# Two Class Datasets
# For all the datasets in the list or one in particular, here for Adult
for i in range(7,8):
    print(DataName[i])
    DATA= '.\\experiments\\'+DataName[i]+'\\'+DataName[i]+'.txt'
    KDIC= '.\\experiments\\'+DataName[i]+'\\'+DataName[i]+'.kdic'
    FOLDER='.\\experiments\\'+DataName[i]

    print(DATA)
    file = pathlib.Path(DATA)
    if file.exists ():
        print ("File exist train start")
        # crete the scenario
        f1 = open('SCENARIO.prm',"w")
        f1.write('\n')
        f1.write('ClassManagement.OpenFile\n')
        f1.write('ClassFileName '  + KDIC + '\n')
        f1.write('OK\n')
        f1.write('AnalysisSpec.PredictorsSpec.ConstructionSpec.MaxTreeNumber 0 \n')
        f1.write('AnalysisSpec.PredictorsSpec.ConstructionSpec.MaxConstructedAttributeNumber 0 \n')
        f1.write('AnalysisSpec.TargetAttributeName ' + Target[i] + '\n')
        f1.write('AnalysisResults.ResultFilesDirectory ' + FOLDER + '\n')
        f1.write('TrainDatabase.DatabaseFiles.List.Key ' +  DataName[i] + '\n')       
        f1.write('TrainDatabase.DatabaseFiles.DataTableName '  + DATA + '\n')
        f1.write('TrainDatabase.SampleNumberPercentage 100.0 \n')
        f1.write('ComputeStats\n')
        f1.write('Exit\n')
        f1.write('OK\n')
        f1.close()
        
        #to know where the bin directory of khiops is :
        subtemp=pk.get_runner().khiops_bin_dir
        #create the line to call the soft
        temp= subtemp + '\khiops.cmd -i SCENARIO.prm -b -e ' + FOLDER + '\\logsTrainModel.txt'
        #train the model
        zz=subprocess.call(temp)
    else:
        print ("File not exist no train")
        
