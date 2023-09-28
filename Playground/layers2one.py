import cv2
import numpy as np

imageCoreTest = '../Images/CT_RETINA/Playground/'
images = []
nLayers = 9
layersNames = ['BCG', 'NFL', 'GCL', 'INL', 'OPL' , 'ONL', 'ELZ', 'RPE', 'CHO'] #layers names 
coreName = 'NORMAL4_'
sU = 640
sV = 640

newImage = np.zeros((sU,sV), dtype = np.uint8)

for j in range(0,nLayers):
    imName = coreName + str(j) + '_' + layersNames[j] +'.jpg'
    img = cv2.imread(imageCoreTest+imName,0) 
    img2 = np.reshape(img, (*img.shape, 1))
    img3 = np.transpose(img2, (2, 0, 1))
    images.append(img3)

for i in range (0, sU):
    for j in range(0, sV):
        mxVect =[0]
        for k in range(1,nLayers):
            mxVect.append(images[k][0, i, j])
        #find max representative index
        mV = np.argmax(mxVect)
        if mV > 0:
            newImage[i,j] = 31*(9-mV) 

cv2.imshow('nimage', newImage)
print('end')