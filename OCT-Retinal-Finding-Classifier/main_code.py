# Convolutional Neural Network

# Importing the libraries
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator
tf.__version__

# Part 1 - Data Preprocessing

# Preprocessing the Training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('/Users/saikoushikmupparapu/Desktop/Intern/Training/eyetype',
                                                 target_size = (256, 256),
                                                 batch_size = 32,
                                                )
input_shape = training_set[0][0].shape
print("Input shape:", input_shape)
# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('/Users/saikoushikmupparapu/Desktop/Intern/Testing',
                                            target_size = (256, 256),
                                            batch_size = 32,
                                           )
# Part 2 - Building the CNN
#print(train_datagen)
# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[256, 256, 3]))

# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=5, activation='sigmoid'))

# Part 3 - Training the CNN
# Import the ModelCheckpoint callback
from keras.callbacks import ModelCheckpoint

# Part 3 - Training the CNN

# Define the filepath where you want to save the best model
# You can change the path and filename as per your requirement
filepath = '/Users/saikoushikmupparapu/Desktop/Intern/best_model.h5'

# Define the ModelCheckpoint callback to save the best model
# The monitor parameter specifies the metric to monitor (val_accuracy in this case)
# The mode parameter specifies whether to maximize ('max') or minimize ('min') the monitored metric
# The save_best_only parameter ensures that only the best model based on validation accuracy is saved
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)

# Compile the CNN
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
# Pass the ModelCheckpoint callback to the callbacks parameter of the fit method
cnn.fit(x=training_set, validation_data=test_set, epochs=100, callbacks=[checkpoint])
# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])




import numpy as np

from tensorflow.keras.preprocessing import image
test_image = image.load_img('/Users/saikoushikmupparapu/Desktop/Intern/Testing/DR24Test/DR85.jpeg', target_size = (256, 256))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
print(result)
print(training_set.class_indices)
print(result[0][0])
if result[0][0] == 1:
    prediction = 'ARI'
elif result[0][1]==1:
    prediction = 'CRI'
elif result[0][2]==1:
    prediction = 'DRI'
elif result[0][3]==1:
    prediction = 'MH1'
else:
    prediction='NR1'

print("22222222")
print(prediction)
test_accuracy = cnn.evaluate(test_set)[1]

# Print the test set accuracy
print("Test set accuracy: {:.2f}%".format(test_accuracy * 100))
# ... (previous code remains unchanged)

# Part 4 - Making predictions on the Test set

# Get the class indices and labels from the training set
class_indices = training_set.class_indices
labels = list(class_indices.keys())

# Get the total number of samples in the test set
total_samples = len(test_set.filenames)

# Generate predictions for the entire test set
predictions = cnn.predict(test_set, steps=total_samples // test_set.batch_size + 1)

# Convert the predictions (probabilities) to binary predictions (0 or 1) based on a threshold (0.5 in this case)
binary_predictions = (predictions > 0.5).astype(int)

# Get the ground truth labels for the test set
ground_truth_labels = test_set.labels

# Print the results for each image in the test set
print("Results for the Test set:")
count=0
for i in range(total_samples):
    image_path = test_set.filepaths[i]
    predicted_label = labels[np.argmax(predictions[i])]
    ground_truth_label = labels[ground_truth_labels[i]]
    if predicted_label==ground_truth_label:
      count=count+1
    
    print("Image: {}, Predicted Label: {}, Ground Truth Label: {}".format(image_path, predicted_label, ground_truth_label))
print(count)