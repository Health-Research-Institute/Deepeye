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
TRAIN_PATH_Y = 'OCT-Retinal-Layer-Segmenter/trainingSet/maskImages/'
train_ids_x = next(os.walk(TRAIN_PATH_X))[2]
train_ids_y = next(os.walk(TRAIN_PATH_Y))[2]

#Capture training image info as a list
train_images = []
for directory_path in glob.glob(TRAIN_PATH_X):
    for img_path in glob.glob(os.path.join(directory_path, "*.jpeg")):
        img = cv2.imread(img_path, 0)       
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        train_images.append(img)    
#Convert list to array for machine learning processing        
train_images = np.array(train_images)

#Capture mask/label info as a list
train_masks = [] 
for directory_path in glob.glob(TRAIN_PATH_Y):
    for mask_path in glob.glob(os.path.join(directory_path, "*.jpeg")):
        mask = cv2.imread(mask_path, 0)       
        mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)  #Otherwise ground truth changes due to interpolation
        train_masks.append(mask)
#Convert list to array for machine learning processing          
train_masks = np.array(train_masks)


def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=SIZE_Y, IMG_WIDTH=SIZE_X, IMG_CHANNELS=1)

model = get_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#model.summary()

model.load_weights('OCT-Retinal-Layer-Segmenter/retina_segmentation_8_layer.hdf5')
#model.load_weights('OCT-Retinal-Layer-Segmenter/retina_segmentation_7_layer_3.hdf5')


#Encode labels
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
n, h, w = train_masks.shape
train_masks_reshaped = train_masks.reshape(-1,1)
train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

np.unique(train_masks_encoded_original_shape)

train_images = np.expand_dims(train_images, axis=3)
train_images = normalize(train_images, axis=1)

train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)


from sklearn.model_selection import train_test_split
# X1, X_test, y1, y_test = train_test_split(train_images, train_masks_input, test_size = 0.10, random_state = 0)

#Further split training data t a smaller subset for quick testing of models
# X_train, X_do_not_use, y_train, y_do_not_use = train_test_split(X1, y1, test_size = 0.2, random_state = 0)

X_train, X_test, y_train, y_test = train_test_split(train_images, train_masks_input, test_size = 0.10, random_state = 0)

# number 7 is now macula hole
test_img_number = 6
test_img = X_train[test_img_number]
ground_truth= y_train[test_img_number]
test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img_norm, 0)

#prediction = (model.predict(test_img_input))
prediction = (model.predict(test_img_input))
predicted_img = np.argmax(prediction, axis=3)[0,:,:]


print('size resized img', img.shape)
print('size of train_images', train_images.shape)
# print('size of X1' , X1.shape)
print('size of X_test', X_train.shape)
print('size of X_train' , X_train.shape)
print('size of test_img' , test_img.shape)
print('size of test_img_norm' , test_img_norm.shape)
print('size of test' , test_img_input.shape)
print('size of prediction' , prediction.shape)
print('size of pred IMAGE' , predicted_img.shape)

plt.figure(figsize=(20, 10))
plt.subplot(261)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0])
plt.subplot(262)
plt.title('Testing Label')
# now lets apply it to just one image 
# testImage=np.expand_dims(0, img, 0)

# plt.imshow(ground_truth[:,:,0], cmap='jet')
# plt.imshow(testImage[0,:,:,0], cmap='jet')


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








