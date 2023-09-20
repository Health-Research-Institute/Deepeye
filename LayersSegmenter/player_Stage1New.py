from simple_unet import multi_unet_model
from keras.utils import normalize
import os
import glob
import cv2
import numpy as np
import pandas as pd

# General Params to use with player
writing = 0 #0 - Train, 1- Test 

sizeX = 640
sizeY = 640
nLayers= 9 # Number of classes for segmentation
layersNames = ['BCG', 'NFL', 'GCL', 'INL', 'OPL' , 'ONL', 'ELZ', 'RPE', 'CHO'] #layers names 
nOrder =  [0, 5, 3, 6, 1, 7, 4, 2, 8] #Sequence of layers per BN model:
classNames = ['AMD','CSR','DR','MH','NORMAL']

modelFolderName = '../Models/segmentModels/retina_segmentation_8_layer.hdf5'
imageCoreFolder = '../Images/CT_RETINA/'

if writing == 0:
    indicesFolder = '../Models/indices/train_indices.csv'
else:
    indicesFolder = '../Models/indices/test_indices.csv'

# LOAD MODEL
model = multi_unet_model(n_classes=nLayers, IMG_HEIGHT=sizeY, IMG_WIDTH=sizeX, IMG_CHANNELS=1)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights(modelFolderName)
# End LOAD MODEL

for tClass in range(0,len(classNames)):
    
    imagesPathRead = imageCoreFolder + classNames[tClass] + '/All'
    if writing == 0:
        imagesTTWrite = imageCoreFolder + classNames[tClass] + '/Train9L'
    else:
        imagesTTWrite = imageCoreFolder + classNames[tClass] + '/Test9L'

    if not os.path.isdir(imagesTTWrite):
        os.makedirs(imagesTTWrite)
    imagesTTWrite = imagesTTWrite + '/'

    imageIds = next(os.walk(imagesPathRead))[2]

    # LOAD FILE INDECES FOR TRAIN or TEST )
    dfTT = pd.read_csv(indicesFolder, low_memory=False) #path to csv file with train indices 

    a1 = dfTT.Names[tClass]
    nIm = dfTT['# Images'][tClass]
    b1 = a1.split('[')
    b2 = b1[1]
    c1 = b2.split(']')
    c2 = c1[0]
    ttInd =c2.split(',')

    # PREPARING IMAGES FOR PLAYER
    initImages = []
    for directoryPath in glob.glob(imagesPathRead):
        for jj in range(0,nIm):
            if jj==0:
                imgPath = directoryPath + '/' + ttInd[jj][1:-1]
            else: 
                imgPath = directoryPath + '/' + ttInd[jj][2:-1] 
            img = cv2.imread(imgPath, 0)       
            img = cv2.resize(img, (sizeY, sizeX))
            initImages.append(img)  
        
    initImages = np.array(initImages) #size is [nImages, sizeX, sizeY]  
    nImages, w, h = initImages.shape
    secImages = np.expand_dims(initImages, axis=3) #size: [nImages, sizeX, sizeY, 1]  
    thirdImages = normalize(secImages, axis=1) #size: [nImages, sizeX, sizeY, 1] 

    # LOOP VIA IMAGES 
    for i in range(0,nIm):
        valImage=thirdImages[i,:,:,:] #size: [sizeX, sizeY, 1] 
        testInput=np.expand_dims(valImage, 0) #size: [1, sizeX, sizeY, 1] 
        prediction = (model.predict(testInput))  #size [1, sizeX, sizeY, 9] 

        if i==0:
            subStr = ttInd[i][1:-1]
        else: 
            subStr =  ttInd[i][2:-1] 
        clNN = subStr[0:-5] #dropping .jpeg from the file name

        #loop through nine layers 
        for j in range(0,nLayers):        
            layer01 = prediction[0,:,:,nOrder[j]] *255
            layerTest = layer01.round()
            #check if it is train or test index and write to the corresponding folder 
            cv2.imwrite(imagesTTWrite + clNN + '_' + str(j) + '_' + layersNames[j] + '.jpg', layerTest)
    
    print('Class',  classNames[tClass], ' is done')




        


