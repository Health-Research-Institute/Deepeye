from simple_unet import multi_unet_model
from keras.utils import normalize
import os
import glob
import cv2
import numpy as np
from layers2one import l2one

# General Params to use with player
sizeX = 640
sizeY = 640
nLayers= 9 # Number of classes for segmentation
layersNames = ['BCG', 'NFL', 'GCL', 'INL', 'OPL' , 'ONL', 'ELZ', 'RPE', 'CHO'] #layers names 
nOrder =  [0, 5, 3, 6, 1, 7, 4, 2, 8] #Sequence of layers per BN model:

#classNames = ['AMD','CSR','DR','MH','OUTLIERS','NORMAL', 'VNORMAL']
classNames = ['OUTLIERS','VNORMAL']
modelDirName = '../Models/segmentModels/retina_segmentation_8_layer.hdf5'

# LOAD MODEL
model = multi_unet_model(n_classes=nLayers, IMG_HEIGHT=sizeY, IMG_WIDTH=sizeX, IMG_CHANNELS=1)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights(modelDirName)
# End LOAD MODEL

for nClass in classNames:
    imagesPathRead = '../Images/CT_RETINA/' + nClass + '/All/'
    imagesPathWrite ='../Images/CT_RETINA/' + nClass + '/All9L/'
    imagesPathWriteS9 = '../Images/CT_RETINA/' + nClass + '/S9L/'
    
    if not os.path.isdir(imagesPathWrite):
        os.makedirs(imagesPathWrite)
    imagesPathWrite = imagesPathWrite + '/'

    if not os.path.isdir(imagesPathWriteS9):
        os.makedirs(imagesPathWriteS9)
    imagesPathWriteS9 = imagesPathWriteS9 + '/'
    
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

        subStr = imageIds[i]
        clNN = subStr[0:-5] #dropping .jpeg from the file name

        imageSt = []
        #loop through nine layers 
        for j in range(0,nLayers):        
            layer01 = prediction[0,:,:,nOrder[j]] *255
            layerTest = layer01.round() 
            cv2.imwrite(imagesPathWrite + clNN + '_' + str(j) + '_' + layersNames[j] + '.jpg', layerTest)
            #create Superposed 
            imageSt.append(layerTest)

        newImage = l2one(imageSt)
        cv2.imwrite(imagesPathWriteS9 + clNN + '_S9.jpg', newImage)    

    print('Class',  nClass, ' part' , ' is done')
