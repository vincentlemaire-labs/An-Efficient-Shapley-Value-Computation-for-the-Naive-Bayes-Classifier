
import os
import subprocess
import time
import pathlib

# khiops should be install before
from pykhiops import core as pk

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
    MODELINGKDIC= '.\\experiments\\'+DataName[i]+'\\Modeling.kdic'

    print(MODELINGKDIC)
    file = pathlib.Path(MODELINGKDIC)
    if file.exists ():
        print ("Modeling File exist so create the files of interpretation (WoE and Shap)")
        # create the scenario
        f1 = open('SCENARIO3.prm',"w")
        # FIRST STEP CREATE THE Interpretation dictionary
        f1.write('\n')
        f1.write('ClassManagement.OpenFile \n')
        # take the trained classifier
        f1.write('ClassFileName ' + MODELINGKDIC + '\n')
        f1.write('OK \n')
        # go into the interpretation menu"
        f1.write('LearningTools.InterpretPredictor \n')
        # specify which interpretation
        f1.write('WhyParameter.WhyNumber ' +  str(NbVar[i]) + '\n')
        f1.write('WhyParameter.WhyClass All classes \n')
        f1.write('WhyParameter.WhyType Shapley \n')
        f1.write('WhyParameter.SortWhy false  \n')
        f1.write('WhyParameter.ExpertMode true \n')
        f1.write('BuildInterpretationClass \n')
        f1.write('ClassFileName ' + FOLDER + '\Interpretation_SHAP.kdic \n')
        f1.write('OK \n')
        f1.write('WhyParameter.WhyType Weight of evidence \n')
        f1.write('BuildInterpretationClass \n')
        f1.write('ClassFileName ' + FOLDER + '\Interpretation_WOE.kdic \n')
        f1.write('OK \n')
        f1.write('Exit \n')
        f1.write('ClassManagement.CloseFile  \n')
        
        # SECOND STEP DEPLOY the Interpretation dictionary
        f1.write('ClassManagement.OpenFile \n')
        f1.write('ClassFileName ' + FOLDER + '\Interpretation_SHAP.kdic \n')
        f1.write('OK \n')
        f1.write('LearningTools.TransferDatabase \n')
        f1.write('SourceDatabase.DatabaseFiles.List.Key Interpretation_SNB_' + DataName[i] + '\n')
        f1.write('SourceDatabase.DatabaseFiles.DataTableName ' + DATA + '\n')
        f1.write('TargetDatabase.DatabaseFiles.List.Key Interpretation_SNB_' + DataName[i] + '\n')
        f1.write('TargetDatabase.DatabaseFiles.DataTableName ' + FOLDER + '\Interpretation_SHAP.txt \n')
        f1.write('TransferDatabase  \n')
        f1.write('Exit \n')
        f1.write('ClassManagement.CloseFile  \n')

        f1.write('ClassManagement.OpenFile \n')
        f1.write('ClassFileName ' + FOLDER + '\Interpretation_WOE.kdic \n')
        f1.write('OK \n')
        f1.write('LearningTools.TransferDatabase \n')
        f1.write('SourceDatabase.DatabaseFiles.List.Key Interpretation_SNB_' + DataName[i] + '\n')
        f1.write('SourceDatabase.DatabaseFiles.DataTableName ' + DATA + '\n')
        f1.write('TargetDatabase.DatabaseFiles.List.Key Interpretation_SNB_' + DataName[i] + '\n')
        f1.write('TargetDatabase.DatabaseFiles.DataTableName ' + FOLDER + '\Interpretation_WOE.txt \n')
        f1.write('TransferDatabase  \n')
        f1.write('Exit \n')
        
        f1.write('Exit \n')
        f1.write('OK \n')
        f1.close()
        
        #train the model
        subtemp=pk.get_runner().khiops_bin_dir
        #create the line to call the soft
        temp= subtemp + '\khiops.cmd -i SCENARIO3.prm -b -e ' + FOLDER + '\\logsInterpretModel.txt'
        zz=subprocess.call(temp)
    else:
        print ("File not exist no interpretation")
        


