import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.layers import Flatten, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import sys
import os

# Part 1 - Data Preprocessing
deepeye_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")) # Folder containing Deepeye repo

# Preprocessing the Training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
training_set = train_datagen.flow_from_directory(
    os.path.join(deepeye_path, 'Images/SaiImages/MultiClassifier/Training/eyetype'),
    target_size=(256, 256),
    batch_size=32
)

# Get the list of training image filenames
training_image_filenames = training_set.filenames

# Save the training image filenames to a file
with open(os.path.join(deepeye_path, 'Logs/SaiLogs/multi_training_image_filenames.txt'), 'w') as file:
    for filename in training_image_filenames:
        file.write(filename + '\n')

# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
    os.path.join(deepeye_path, 'Images/SaiImages/MultiClassifier/Testing'),
    target_size=(256, 256),
    batch_size=32
)

# Part 2 - Building the Transfer Learning Model
# Load the VGG16 model without the top (fully connected) layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Adding custom layers for classification
x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(5, activation='sigmoid')(x)

# Create the transfer learning model
tl_model = Model(inputs=base_model.input, outputs=output)

# Compile the model
tl_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display model summary
tl_model.summary()

# Part 3 - Training the Transfer Learning Model
# Define the filepath where you want to save the best model
filepath = os.path.join(deepeye_path, 'Models/SaiModels/best_tl_multi_model.h5')

# Define the ModelCheckpoint callback
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)

# Training the transfer learning model
history = tl_model.fit(x=training_set, validation_data=test_set, epochs=20, callbacks=[checkpoint])




plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(deepeye_path, 'Figures/SaiFigures/multi_training_accuracy_plot.png'))
plt.show()

# Save the validation accuracy plot
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(deepeye_path, 'Figures/SaiFigures/multi_validation_accuracy_plot.png'))
plt.show()

# Save the output logs
sys.stdout = open(os.path.join(deepeye_path, 'Logs/SaiLogs/multi_output_logs.txt'), 'w')
print(history.history)
sys.stdout.close()


# Evaluate the model on the test set
#test_loss, test_accuracy = tl_model.evaluate(test_set)
#print("Test set accuracy: {:.2f}%".format(test_accuracy * 100))
