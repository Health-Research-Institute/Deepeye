import os
import numpy as np
import pandas as pd
from torch.utils.data import random_split
import datetime


dataset = 'OCTID'
#dataset = 'KAGLE'
percSplit = 0.825

if dataset == 'OCTID':
    classNames = ['AMD','DR', 'CSR','MH','OUTLIERS', 'NORMAL', 'VNORMAL']
    nTrain = [60, 60, 60, 60, 30, 160, 16]
    imageCoreDir = '../Images/CT_RETINA'
elif dataset == 'KAGLE':
    classNames = ['CNV','DME','DRUSEN','NORMAL']
    nTrain = [50, 50, 50, 150]
    imageCoreDir = '../../../KagleDataSet/test'
nSickClasses = len(classNames) - 1


logsDir = '../Logs/TrainTestNames/'

current_time= datetime.datetime.now()
fullTime= str(current_time)
current_date = fullTime[:10] + '-' + fullTime[11:13] + '-' + fullTime[14:16]

trainIndName = logsDir + dataset + '_' + str(current_date) + 'train.csv'
testIndName = logsDir + dataset + '_'  + str(current_date) + 'test.csv'
valIndName = logsDir + dataset + '_'  + str(current_date) + 'val.csv'

dfTrain = pd.DataFrame(columns=['# Images', 'Class', 'Names'])
dfTest =  dfTrain
dfVal =  dfTrain

iClass = 0
for nClass in classNames:
    if dataset == 'OCTID':
        imagesPathRead = imageCoreDir + '/' + nClass + '/All'
    elif dataset == 'KAGLE':
        imagesPathRead = imageCoreDir + '/' + nClass

    file_names = os.listdir(imagesPathRead)
    nFiles = len(file_names)
    indArray = np.arange(0, nFiles)

    nImClass = nTrain[iClass]
    
    if (percSplit* nFiles < nImClass):
        nT = round(percSplit*nFiles)
        train_set, test_set = random_split(indArray, [nT, nFiles-nT])
    else:
        train_set, test_set = random_split(indArray, [nImClass, nFiles-nImClass])
   
    #split train set to train and valuation
    lenTrain = len(train_set)
    valN = round((1-percSplit)*lenTrain)
    train_set, val_set = random_split(train_set, [lenTrain-valN, valN])


    trainInd = np.sort(train_set).tolist()
    testInd = np.sort(test_set).tolist()
    valInd = np.sort(val_set).tolist()

    namesTr = [file_names[i] for i in trainInd]
    namesTe = [file_names[i] for i in testInd]
    namesVa = [file_names[i] for i in valInd]

    dfTrain = pd.concat([dfTrain, pd.DataFrame.from_records([{'# Images': len(namesTr), 'Class': nClass,  'Names': namesTr}])], ignore_index=True)
    dfTrain.to_csv(trainIndName, index=False)

    dfVal = pd.concat([dfVal, pd.DataFrame.from_records([{'# Images': len(namesVa), 'Class': nClass,  'Names': namesVa}])], ignore_index=True)
    dfVal.to_csv(valIndName, index=False)


    dfTest  = pd.concat([dfTest, pd.DataFrame.from_records([{'# Images': len(namesTe), 'Class': nClass, 'Names': namesTe}])], ignore_index=True)
    dfTest.to_csv(testIndName , index=False)

    iClass = iClass+1

print('Created three lists in TrainTestNames')   