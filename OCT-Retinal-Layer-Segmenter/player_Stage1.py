from simple_unet import multi_unet_model
from keras.utils import normalize
import os
import glob
import cv2
import numpy as np
import sys

# General Params to use with player
testMode = 'False'

sizeX = 640
sizeY = 640
nLayers= 9 # Number of classes for segmentation
layersNames = ['BCG', 'NFL', 'GCL', 'INL', 'OPL' , 'ONL', 'ELZ', 'RPE', 'CHO'] #layers names 
nOrder =  [0, 5, 3, 6, 1, 7, 4, 2, 8] #Sequence of layers per BN model:

classNames = ['AMRD','CSR','DR','MH','NORMAL']

# LOAD MODEL
model = multi_unet_model(n_classes=nLayers, IMG_HEIGHT=sizeY, IMG_WIDTH=sizeX, IMG_CHANNELS=1)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights('OCT-Retinal-Layer-Segmenter/retina_segmentation_8_layer.hdf5')
# End LOAD MODEL


for testClass in range(0,5):
    
    if (testMode == 'False'):
        folderN = 'Train'
        addName = 0
    else:  
        folderN = 'Test'
        if testClass == 4:
            addName = 54
        else:
            addName = 18 

    imagesPathRead = '../Images/CT_RETINA/' + classNames[testClass] + '/' + folderN
    imagesPathWrite ='../Images/CT_RETINA/' + classNames[testClass] + '/' + folderN + '9L'
    if not os.path.isdir(imagesPathWrite):
        os.makedirs(imagesPathWrite)
    imagesPathWrite = imagesPathWrite + '/'
    
    imageIds = next(os.walk(imagesPathRead))[2]

    # PREPARING IMAGES FOR PLAYER
    initImages = []
    for directoryPath in glob.glob(imagesPathRead):
        for imgPath in glob.glob(os.path.join(directoryPath, "*.jpeg")):
            img = cv2.imread(imgPath, 0)       
            img = cv2.resize(img, (sizeY, sizeX))
            initImages.append(img)  
        
    initImages = np.array(initImages) #size is [nImages, sizeX, sizeY]  
    nImages, w, h = initImages.shape
    secImages = np.expand_dims(initImages, axis=3) #size: [nImages, sizeX, sizeY, 1]  
    thirdImages = normalize(secImages, axis=1) #size: [nImages, sizeX, sizeY, 1] 

    # LOOP VIA IMAGES 
    for i in range(0,len(imageIds)):
        valImage=thirdImages[i,:,:,:] #size: [sizeX, sizeY, 1] 
        testInput=np.expand_dims(valImage, 0) #size: [1, sizeX, sizeY, 1] 
        prediction = (model.predict(testInput))  #size [1, sizeX, sizeY, 9] 
        predicted_img = np.argmax(prediction, axis=3)[0,:,:] #size: [sizeX, sizeY] 

        #loop through nine layers 
        for j in range(0,9):        
            layer01 = prediction[0,:,:,nOrder[j]] *255
            layerTest = layer01.round() 
            cv2.imwrite(imagesPathWrite + classNames[testClass] + str (i+1+addName) + '_' + str(j) + '_' + layersNames[j] + '.jpg', layerTest)
    
    print('Class',  testClass, ' is done')
