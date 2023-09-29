import cv2
import numpy as np


def imageStack (coreDir, coreName):
    nLayers = 9
    layersNames = ['BCG', 'NFL', 'GCL', 'INL', 'OPL' , 'ONL', 'ELZ', 'RPE', 'CHO']   
    imageSt = []
    for j in range(0,nLayers):
        imName = coreName + str(j) + '_' + layersNames[j] +'.jpg'
        img = cv2.imread(coreDir+imName,0) 
        img2 = np.reshape(img, (*img.shape, 1))
        img3 = np.transpose(img2, (2, 0, 1))
        imageSt.append(img3)

    return imageSt


def l2one(imageSt):

    nLayers = 9
    sU = imageSt[0].shape[0]
    sV = imageSt[0].shape[1]
    newImage = np.zeros((sU,sV), dtype = np.uint8)

    for i in range (0, sU):
        for j in range(0, sV):
            mxVect =[0]
            for k in range(1,nLayers):
                mxVect.append(imageSt[k][i, j])
            #find max representative index
            mV = np.argmax(mxVect)
            if (mV > 0) and (mxVect[mV] > 16):
                newImage[i,j] = 31*(9-mV) 

    return newImage