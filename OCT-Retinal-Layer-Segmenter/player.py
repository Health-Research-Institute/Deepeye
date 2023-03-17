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
# Image directory
# imagesPath = 'OCT-Retinal-Layer-Segmenter/trainingSet/origImages/'
imagesPath = '../Images/CT_RETINA/DIABETR_107/'
imageIds = next(os.walk(imagesPath))[2]
#Parameters to use with player
sizeX = 640
sizeY = 640
nClasses= 9 # Number of classes for segmentation
# End PARAMETERS

# PREPARING IMAGES FOR PLAYER
initImages = []
for directoryPath in glob.glob(imagesPath):
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

# LOOP VIA IMAGES 
#for i in range(nImages):
for i in range(0,8):
    valImage=thirdImages[i,:,:,:] #size: [sizeX, sizeY, 1] 
    testInput=np.expand_dims(valImage, 0) #size: [1, sizeX, sizeY, 1] 
    prediction = (model.predict(testInput))  #size [1, sizeX, sizeY, 9] 
    predicted_img = np.argmax(prediction, axis=3)[0,:,:] #size: [sizeX, sizeY] 

    # PLOT INFORMATION
    plt.figure(figsize=(20, 10))
    plt.subplot(261)
    plt.title('Testing Image')
    plt.imshow(valImage[:,:,0])
    plt.subplot(262)
    plt.imshow(prediction[0,:,:,0], cmap='jet')
    plt.title('Background Predicition ')
    plt.subplot(263)
    plt.imshow(prediction[0,:,:,5], cmap='jet')
    plt.subplot(264)
    plt.imshow(prediction[0,:,:,3], cmap='jet')
    plt.subplot(265)
    plt.imshow(prediction[0,:,:,6], cmap='jet')
    plt.subplot(266)
    plt.imshow(prediction[0,:,:,1], cmap='jet')
    plt.subplot(267)
    plt.imshow(prediction[0,:,:,7], cmap='jet')
    plt.subplot(268)
    plt.imshow(prediction[0,:,:,4], cmap='jet')
    plt.subplot(269)
    plt.imshow(prediction[0,:,:,2], cmap='jet')
    plt.subplot(2,6,10)
    plt.imshow(prediction[0,:,:,8], cmap='jet')
    plt.subplot(2,6,11)
    plt.title('Combines Prediction')
    plt.imshow(predicted_img, cmap='jet')
    plt.show()