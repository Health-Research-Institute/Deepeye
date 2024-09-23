import os
import numpy as np
import shutil
import pandas as pd
from torch.utils.data import random_split
import datetime
import cv2

#Helper function to clean old files
def delete_files_and_subdir(directory_path):
   try:
     with os.scandir(directory_path) as entries:
       for entry in entries:
         if entry.is_file():
            os.unlink(entry.path)
         else:
            shutil.rmtree(entry.path)
     print("All files in " + directory_path + " deleted successfully.")
   except OSError:
     print("Error occurred while deleting files and subdirectories.")

###### SET_UP PARAMETERS
# Specify the partition log file from where image file names should be taken
trainLogFile = 'KAGLE_2024-09-18-18-27train.csv'

#Extraxt data set name
dataset = trainLogFile.split('_')[0] 

recordDate = trainLogFile.split('t')[0] 
testLogFile =  recordDate + 'test.csv'
valLogFile =  recordDate + 'val.csv'
logsFromDir = '../Logs/TrainTestNames/'
# Directory from where images are taken
# 
if dataset == 'OCTID': 
    imageFromDir = '../Images/CT_RETINA'
elif dataset == 'KAGLE':
    imageFromDir = '../../../KagleDataSet/test'

#Destination directories where to write files 
imageToCore = '../Images/CT_RETINA_BinaryResearch'
imageToDirCNN = imageToCore  + '/cnnBinary'
imageToDirDense = imageToCore  + '/denseBinary'
algs = [imageToDirCNN, imageToDirDense]
layersNames = ['_0_BCG', '_1_NFL', '_2_GCL', '_3_INL', '_4_OPL' , '_5_ONL', '_6_ELZ', '_7_RPE', '_8_CHO']

#CLEAN destination directory from old files and directories 
delete_files_and_subdir(imageToCore)

#write index file
logFile = open(imageToCore +'/dateIndex.txt', "w")
logFile.write(recordDate)
logFile.close()

#Create new difectory structure 
for alg in algs: 
    os.makedirs(alg)
    os.makedirs(alg+ '/Testing')
    os.makedirs(alg + '/Testing/ER')
    os.makedirs(alg + '/Testing/NR')
    os.makedirs(alg + '/Training')
    os.makedirs(alg + '/Training/ER')
    os.makedirs(alg + '/Training/NR')
    os.makedirs(alg + '/Valuation')
    os.makedirs(alg + '/Valuation/ER')
    os.makedirs(alg + '/Valuation/NR')

#TRAINING FILES COPYING
#read line by line:
with open(logsFromDir  + trainLogFile ) as trainF:
    while True:
        line = trainF.readline()
        if not line:
            break
        #ignore first line with titles 
        if len(line) > 40: #proceed with parcing lines 
            noSpace = line.split(",") #split line into image # and names 
            nImages = int(noSpace[0]) #number of images or imagesx8 to copy

            baseClassName = noSpace[1]
            #deal with CNN first, Dense Next
            if dataset == 'OCTID': 
                dirFromCNN = imageFromDir + '/' + baseClassName + '/' + 'All' #put /
                dirFromDense = imageFromDir + '/' + baseClassName + '/' + 'All9L' #put /
            elif dataset == 'KAGLE': 
                dirFromCNN = imageFromDir + '/' + baseClassName


            if baseClassName == 'NORMAL':
                dirToCNN = imageToDirCNN + '/Training/NR'
                dirToDense = imageToDirDense + '/Training/NR'
            else:
                dirToCNN = imageToDirCNN + '/Training/ER'
                dirToDense = imageToDirDense + '/Training/ER'
        
            # copy images  
            for i in range(0, nImages):
                #extract base file name
                #NOTICE to skip 2 indexes in noSpace array
                if dataset == 'OCTID': 
                    base_file_name = ''.join([c for c in noSpace[i+2] if c.isupper() or c.isdigit()])             
                elif dataset == 'KAGLE': 
                    base_file_name = ''.join([c for c in noSpace[i+2] if c.isupper() or c.isdigit() or c=='-'])
                #add .jpeg for cnn files
                cnnFileName = base_file_name + '.jpeg'
                shutil.copy(os.path.join(dirFromCNN, cnnFileName), dirToCNN)

                if dataset == 'OCTID': 
                    for j in layersNames:
                        denseFileName = base_file_name + j + '.jpg'
                        shutil.copy(os.path.join(dirFromDense, denseFileName), dirToDense)       
trainF.close()

#Testing FILES COPYING

#read line by line:
with open(logsFromDir  + testLogFile ) as testF:
    while True:
        line = testF.readline()
        if not line:
            break
        #ignore first line with titles 
        if len(line) > 40: #proceed with parcing lines 
            noSpace = line.split(",") #split line into image # and names 
            nImages = int(noSpace[0]) #number of images or imagesx8 to copy

            baseClassName = noSpace[1]
            #deal with CNN first, Dense Next
            if dataset == 'OCTID': 
                dirFromCNN = imageFromDir + '/' + baseClassName + '/' + 'All' #put /
                dirFromDense = imageFromDir + '/' + baseClassName + '/' + 'All9L' #put /
            elif dataset == 'KAGLE': 
                dirFromCNN = imageFromDir + '/' + baseClassName

            if (baseClassName == 'NORMAL'):
                dirToCNN = imageToDirCNN + '/Testing/NR'
                dirToDense = imageToDirDense + '/Testing/NR'
            else:
                dirToCNN = imageToDirCNN + '/Testing/ER'
                dirToDense = imageToDirDense + '/Testing/ER'
        
            # copy images  
            for i in range(0, nImages):
                #extract base file name
                #NOTICE to skip 2 indexes in noSpace array
                if dataset == 'OCTID': 
                    base_file_name = ''.join([c for c in noSpace[i+2] if c.isupper() or c.isdigit()])             
                elif dataset == 'KAGLE': 
                    base_file_name = ''.join([c for c in noSpace[i+2] if c.isupper() or c.isdigit() or c=='-'])

                #add .jpeg for cnn files
                cnnFileName = base_file_name + '.jpeg'
                shutil.copy(os.path.join(dirFromCNN, cnnFileName), dirToCNN)

                if dataset == 'OCTID': 
                    for j in layersNames:
                        denseFileName = base_file_name + j + '.jpg'
                        shutil.copy(os.path.join(dirFromDense, denseFileName), dirToDense)       
testF.close()



with open(logsFromDir  + valLogFile ) as valF:
    while True:
        line = valF.readline()
        if not line:
            break
        #ignore first line with titles 
        if len(line) > 40: #proceed with parcing lines 
            noSpace = line.split(",") #split line into image # and names 
            nImages = int(noSpace[0]) #number of images or imagesx8 to copy

            baseClassName = noSpace[1]
            #deal with CNN first, Dense Next
            if dataset == 'OCTID': 
                dirFromCNN = imageFromDir + '/' + baseClassName + '/' + 'All' #put /
                dirFromDense = imageFromDir + '/' + baseClassName + '/' + 'All9L' #put /
            elif dataset == 'KAGLE': 
                dirFromCNN = imageFromDir + '/' + baseClassName

            if (baseClassName == 'NORMAL'):
                dirToCNN = imageToDirCNN + '/Valuation/NR'
                dirToDense = imageToDirDense + '/Valuation/NR'
            else:
                dirToCNN = imageToDirCNN + '/Valuation/ER'
                dirToDense = imageToDirDense + '/Valuation/ER'
        
            # copy images  
            for i in range(0, nImages):
                #extract base file name
                #NOTICE to skip 2 indexes in noSpace array
                if dataset == 'OCTID': 
                    base_file_name = ''.join([c for c in noSpace[i+2] if c.isupper() or c.isdigit()])             
                elif dataset == 'KAGLE': 
                    base_file_name = ''.join([c for c in noSpace[i+2] if c.isupper() or c.isdigit() or c=='-'])

                #add .jpeg for cnn files
                cnnFileName = base_file_name + '.jpeg'
                shutil.copy(os.path.join(dirFromCNN, cnnFileName), dirToCNN)

                if dataset == 'OCTID': 
                    for j in layersNames:
                        denseFileName = base_file_name + j + '.jpg'
                        shutil.copy(os.path.join(dirFromDense, denseFileName), dirToDense)       
valF.close()






print('Files were Copied')


