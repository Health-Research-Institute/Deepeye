import os
import numpy as np
# import shutil
import pandas as pd
from torch.utils.data import random_split
import datetime

#classNames = ['AMD','CSR','DR','MH','NORMAL']
classNames = ['AMD','DR','MH','NORMAL']
nSickClasses = len(classNames) - 1
nTrain = 75 #for Normal this will be multiplied for number of classes
imageCoreDir = '../Images/CT_RETINA'
logsDir = '../Logs/TrainTestNames/'

current_time= datetime.datetime.now()
fullTime= str(current_time)
current_date = fullTime[:10] + '-' + fullTime[11:13] + '-' + fullTime[14:16]


trainIndName = logsDir + 'trainNames_' + str(current_date) + '.csv'
testIndName = logsDir + 'testNames_' + str(current_date) + '.csv'

dfTrain = pd.DataFrame(columns=['# Images', 'Names'])
dfTest =  pd.DataFrame(columns=['# Images', 'Names'])


for nClass in classNames:
    imagesPathRead = imageCoreDir + '/' + nClass + '/All'
    
    #imagesPathWriteTrain = imageCoreDir + '/CNNTrain/' + nClass
    #if not os.path.isdir(imagesPathWriteTrain):
    #    os.makedirs(imagesPathWriteTrain)
    #imagesPathWriteTrain = imagesPathWriteTrain + '/'

    #imagesPathWriteTest = imageCoreDir + '/CNNTest/' + nClass
    #if not os.path.isdir(imagesPathWriteTest):
    #    os.makedirs(imagesPathWriteTest)
    #imagesPathWriteTest = imagesPathWriteTest + '/'

    file_names = os.listdir(imagesPathRead)
    nFiles = len(file_names)
    indArray = np.arange(0, nFiles)
    
    if (nClass == 'NORMAL'): #normal class to have more images
        if (0.8* nFiles < nTrain *nSickClasses):
            nT = round(0.8*nFiles)
            train_set, test_set = random_split(indArray, [nT, nFiles-nT])
        else:
            train_set, test_set = random_split(indArray, [nSickClasses*nTrain, nFiles-nSickClasses*nTrain])
    else:
        if (0.8* nFiles < nTrain):
            nT = round(0.8*nFiles)
            train_set, test_set = random_split(indArray, [nT, nFiles-nT])
        else:
            train_set, test_set = random_split(indArray, [nTrain, nFiles-nTrain])
   
    trainInd = np.sort(train_set).tolist()
    testInd = np.sort(test_set).tolist()

    namesTr = [file_names[i] for i in trainInd]
    namesTe = [file_names[i] for i in testInd]
   
  #  for i in range(0, len(trainInd)):
  #      file_name = file_names[trainInd[i]] 
  #      shutil.copy(os.path.join(imagesPathRead, file_name), imagesPathWriteTrain)

   # for j in range(0, len(testInd)):
   #     file_name = file_names[testInd[j]] 
   #     shutil.copy(os.path.join(imagesPathRead, file_name), imagesPathWriteTest)

    dfTrain = pd.concat([dfTrain, pd.DataFrame.from_records([{'# Images': len(namesTr), 'Names': namesTr}])], ignore_index=True)
    dfTest  = pd.concat([dfTest, pd.DataFrame.from_records([{'# Images': len(namesTe), 'Names': namesTe}])], ignore_index=True)
       
    dfTrain.to_csv(trainIndName, index=False)
    dfTest.to_csv(testIndName , index=False)

print('Created two lists in TrainTestNames')   