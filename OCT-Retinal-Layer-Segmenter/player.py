# from simple_unet import multi_unet_model2
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

#Resizing images
SIZE_X = 640
SIZE_Y = 640
n_classes= 9 # Number of classes for segmentation
TRAIN_PATH_X = 'OCT-Retinal-Layer-Segmenter/trainingSet/origImages/'
train_ids_x = next(os.walk(TRAIN_PATH_X))[2]

#Capture training image info as a list
train_images = []
for directory_path in glob.glob(TRAIN_PATH_X):
    for img_path in glob.glob(os.path.join(directory_path, "*.jpeg")):
        img = cv2.imread(img_path, 0)       
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        train_images.append(img) 
        print('Train image size', len(train_images))   
#Convert list to array for machine learning processing        
train_images = np.array(train_images)

def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=SIZE_Y, IMG_WIDTH=SIZE_X, IMG_CHANNELS=1)

model = get_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#model.summary()

model.load_weights('OCT-Retinal-Layer-Segmenter/retina_segmentation_8_layer.hdf5')

#Encode labels
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
n, h, w = train_images.shape
train_images = np.expand_dims(train_images, axis=3)
train_images = normalize(train_images, axis=1)


test_img_number = 7

test_img_inputOther=train_images[test_img_number,:,:,:]
test_img_input0=np.expand_dims(test_img_inputOther, 0)
#prediction = (model.predict(test_img_input))
prediction = (model.predict(test_img_input0))
predicted_img = np.argmax(prediction, axis=3)[0,:,:]


print('size resized img', img.shape)
print('size of train_images', train_images.shape)
print('size of test_image_inputOther' , test_img_inputOther.shape)
print('size of prediction' , prediction.shape)
print('size of pred IMAGE' , predicted_img.shape)

plt.figure(figsize=(20, 10))
plt.subplot(261)
plt.title('Testing Image')
#plt.imshow(test_img[:,:,0])
plt.subplot(262)
plt.title('Testing Label')

plt.subplot(263)
plt.title('Prediction on test image')
plt.imshow(predicted_img, cmap='jet')

plt.subplot(264)
plt.imshow(prediction[0,:,:,1], cmap='jet')
plt.subplot(265)
plt.imshow(prediction[0,:,:,2], cmap='jet')
plt.subplot(266)
plt.imshow(prediction[0,:,:,3], cmap='jet')
plt.subplot(267)
plt.imshow(prediction[0,:,:,4], cmap='jet')
plt.subplot(268)
plt.imshow(prediction[0,:,:,5], cmap='jet')
plt.subplot(269)
plt.imshow(prediction[0,:,:,6], cmap='jet')
plt.subplot(2,6,10)
plt.imshow(prediction[0,:,:,7], cmap='jet')
plt.subplot(2,6,11)
plt.imshow(prediction[0,:,:,8], cmap='jet')
plt.subplot(2,6,12)
plt.imshow(prediction[0,:,:,0], cmap='jet')

plt.show()








