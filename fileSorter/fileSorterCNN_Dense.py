import os
import numpy as np
import shutil
import pandas as pd
from torch.utils.data import random_split
import datetime

classNames = ['AMD','CSR','DR','MH','NORMAL']
nClasses = len(classNames)
pTrain = 80 #Percentage of images going to training set 
imageCoreDir = '../Images/CT_RETINA'
logsDir = '../Logs/TrainTestNames/'

nLayers= 9 # Number of classes for segmentation
layersNames = ['BCG', 'NFL', 'GCL', 'INL', 'OPL' , 'ONL', 'ELZ', 'RPE', 'CHO'] #layers names 

current_date = datetime.date.today()

trainIndName = logsDir + 'train_Names' + str(pTrain) + '%_' + str(current_date) + '.csv'
testIndName = logsDir + 'test_Names' + str(pTrain) + '%_' + str(current_date) + '.csv'

dfTrain = pd.DataFrame(columns=['# Images', 'Names'])
dfTest =  pd.DataFrame(columns=['# Images', 'Names'])


for nClass in classNames:
    imagesPathRead = imageCoreDir + '/' + nClass + '/All'
    imagesPathRead9L = imageCoreDir + '/' + nClass + '/All9L'
    
    imagesCNNTrain = imageCoreDir + '/TempCNNTrain/' + nClass
    if not os.path.isdir(imagesCNNTrain):
        os.makedirs(imagesCNNTrain)
    imagesCNNTrain = imagesCNNTrain + '/'

    imagesCNNTest = imageCoreDir + '/TempCNNTest/' + nClass
    if not os.path.isdir(imagesCNNTest):
        os.makedirs(imagesCNNTest)
    imagesCNNTest = imagesCNNTest + '/'

    imagesDenseTrain = imageCoreDir + '/TempDenseTrain9L/' + nClass
    if not os.path.isdir(imagesDenseTrain):
        os.makedirs(imagesDenseTrain)
    imagesDenseTrain = imagesDenseTrain + '/'

    imagesDenseTest = imageCoreDir + '/TempDenseTest9L/' + nClass
    if not os.path.isdir(imagesDenseTest):
        os.makedirs(imagesDenseTest)
    imagesDenseTest = imagesDenseTest + '/'

    file_names = os.listdir(imagesPathRead)
    nFiles = len(file_names)
    indArray = np.arange(0, nFiles)

    nTest = round(nFiles*(100-pTrain)/100)
    
    #if (nClass == 'NORMAL'): #normal class to have more images
    #    train_set, test_set = random_split(indArray, [3*nTrain, nFiles-3*nTrain])
    #else:
    train_set, test_set = random_split(indArray, [nFiles - nTest, nTest])
   
    trainInd = np.sort(train_set).tolist()
    testInd = np.sort(test_set).tolist()

    namesTr = [file_names[i] for i in trainInd]
    namesTe = [file_names[i] for i in testInd]
   
    for i in range(0, len(trainInd)):
        file_name = file_names[trainInd[i]] 
        shutil.copy(os.path.join(imagesPathRead, file_name), imagesCNNTrain)
        for j in range(0,nLayers):
            fCore = file_name[0:-5]
            file9L_name = fCore + '_' + str(j) + '_' + layersNames[j] + '.jpg'
            shutil.copy(os.path.join(imagesPathRead9L, file9L_name), imagesDenseTrain)

    for i in range(0, len(testInd)):
        file_name = file_names[testInd[i]] 
        shutil.copy(os.path.join(imagesPathRead, file_name), imagesCNNTest)
        for j in range(0,nLayers):
            fCore = file_name[0:-5]
            file9L_name = fCore + '_' + str(j) + '_' + layersNames[j] + '.jpg'
            shutil.copy(os.path.join(imagesPathRead9L, file9L_name), imagesDenseTest)

    dfTrain = pd.concat([dfTrain, pd.DataFrame.from_records([{'# Images': len(namesTr), 'Names': namesTr}])], ignore_index=True)
    dfTest  = pd.concat([dfTest, pd.DataFrame.from_records([{'# Images': len(namesTe), 'Names': namesTe}])], ignore_index=True)
       
    dfTrain.to_csv(trainIndName, index=False)
    dfTest.to_csv(testIndName , index=False)

print('Moving Files from ALL to Train is done')   