from simple_unet import multi_unet_model
from keras.utils import normalize
import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import random

np.set_printoptions(threshold=sys.maxsize)


# PARAMETERS 

plotFlag = 1

# Image directories
imCoreName = 'MACHOLE' 
imForTrain = 80
imagePathThis  =  imCoreName + '_102/MH1-' + str(imForTrain)

imagePathBase = '../Images/CT_RETINA/'
imagesPathRead = imagePathBase + imagePathThis 
imagesPathWrite = imagesPathRead +'_9Levels/'

imageIds = next(os.walk(imagesPathRead))[2]
#Parameters to use with player
sizeX = 640
sizeY = 640
nClasses= 9 # Number of classes for segmentation
#layers names 
layersNames = ['BCG', 'NFL', 'GCL', 'INL', 'OPL' , 'ONL', 'ELZ', 'RPE', 'CHO']

# End PARAMETERS

# PREPARING IMAGES FOR PLAYER
initImages = []
for directoryPath in glob.glob(imagesPathRead):
    for imgPath in glob.glob(os.path.join(directoryPath, "*.jpeg")):
        img = cv2.imread(imgPath, 0)       
        img = cv2.resize(img, (sizeY, sizeX))
        initImages.append(img)  
     
initImages = np.array(initImages) #size is [nImages, sizeX, sizeY]  
nImages, w, h = initImages.shape
print( 'number of images', nImages)
secImages = np.expand_dims(initImages, axis=3) #size: [nImages, sizeX, sizeY, 1]  
thirdImages = normalize(secImages, axis=1) #size: [nImages, sizeX, sizeY, 1] 
# End PREPARING 

# LOAD MODEL
model = multi_unet_model(n_classes=nClasses, IMG_HEIGHT=sizeY, IMG_WIDTH=sizeX, IMG_CHANNELS=1)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#model.summary()
model.load_weights('OCT-Retinal-Layer-Segmenter/retina_segmentation_8_layer.hdf5')
# End LOAD MODEL

#Sequence of layers per model:
nOrder =  [0, 5, 3, 6, 1, 7, 4, 2, 8]
# The number of corresponding areas (blobs) per layer from top to bottom
# [2 var 2 2 2 1 1 1 1] (first upper layer is var)
nFeatures = np.array([2, 3, 2, 2, 2, 1, 1, 1, 1])
nFeaturesInOrder = nFeatures[nOrder]
#print(nFeaturesInOrder)

#set-up blob detector
# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector()

# LOOP VIA IMAGES 
#for i in range(nImages):
for i in range(0,imForTrain):
    valImage=thirdImages[i,:,:,:] #size: [sizeX, sizeY, 1] 
    testInput=np.expand_dims(valImage, 0) #size: [1, sizeX, sizeY, 1] 
    prediction = (model.predict(testInput))  #size [1, sizeX, sizeY, 9] 
    predicted_img = np.argmax(prediction, axis=3)[0,:,:] #size: [sizeX, sizeY] 

    # PLOT INFORMATION
    plt.figure(figsize=(20, 10))
    plt.subplot(261)
    plt.title('Testing Image')
    plt.imshow(valImage[:,:,0])

    #loop through nine layers 
    for j in range(0,9):
        # Detect blobs.
        
        layer01 = prediction[0,:,:,nOrder[j]] *255
        layerTest = layer01.round() 
        #print(layerTest.min(), ' ... ', layerTest.max())
        cv2.imwrite(imagesPathWrite + imCoreName + str (i+1) + '_' + str(j) + '_' + layersNames[j] + '.jpg', layerTest)
        if plotFlag:
            imBD = cv2.imread(imagesPathWrite + imCoreName + str (i) + '_' + str(j) + '_' + layersNames[j] + '.jpg', cv2.IMREAD_GRAYSCALE)
            plt.subplot(2,6,j+2)
            plt.imshow(prediction[0,:,:,nOrder[j]], cmap='gray')
            #plt.title('Level: ', str(j))
    
    if plotFlag:
        plt.subplot(2,6,11)
        plt.title('Combines Prediction')
        plt.imshow(predicted_img, cmap='jet')
        plt.show()