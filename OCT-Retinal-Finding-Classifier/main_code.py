import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define image preprocessing parameters
preprocess_params = {
     # Normalize pixel values to [0, 1]
    # Add additional preprocessing parameters as needed
}

# Define the CNN model
model = tf.keras.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 4)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))





# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define the function to load images from a folder
def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)  # Convert BGR to RGB
            img = cv2.resize(img, (64, 64))  # Resize image if necessary
            images.append(img)
    return images

# Prepare the data
x_col = []  # List to store image packs
y_col = []  # List to store corresponding labels
eye_folderpath ='/Users/saikoushikmupparapu/Desktop/Intern/Training/eyetype/'
skin_folderpath='/Users/saikoushikmupparapu/Desktop/Intern/Training/skinimage/'
# Iterate over each disease type
eye_fold_name=["AR1-45","CR1-80","DR1-83","MH1-80","NR1-160"]
skin_fold_name=["AR1-45_9","CR1-80_9","DR1-83_9","MH1-80_9","NR1-160_9"]
for disease_type in range(0, 5):
    # Load eye images for the disease type
    eye_folder_path = eye_folderpath+eye_fold_name[disease_type]
    eye_images = load_images_from_folder(eye_folder_path)

    # Load skin images for the disease type
    skin_folder_path = skin_folderpath+skin_fold_name[disease_type]
    skin_images = load_images_from_folder(skin_folder_path)

    # Iterate over each eye image
    




    for i in range(len(eye_images)):
        eye_image = eye_images[i]
        skin_pack = skin_images[i * 9:(i + 1) * 9]  # Extract the corresponding 9 skin images
        eye_image = np.expand_dims(eye_image, axis=-1)

        skin_pack = [np.expand_dims(img, axis=-1) for img in skin_pack]
        image_pack = np.concatenate([eye_image, *skin_pack], axis=-1)
        image_pack = np.expand_dims(image_pack, axis=0)  # Add an extra dimension for the batch size
        x_col.append(image_pack)

        # Create the corresponding label for the image pack
        label = disease_type  # Assuming disease types start from 1, adjust accordingly if needed
        y_col.append(label)
        #print(label)
x_col = np.concatenate(x_col, axis=0)
y_col = np.array(y_col)

print("Shape of x_col:", x_col.shape)
print("Shape of y_col:", y_col.shape)
y_col = np.repeat(y_col, 10) 
# Convert the lists to NumPy arrays
num_classes = 5
x_col = np.transpose(x_col, (0, 1, 2, 4, 3))  # Transpose the dimensions to match the expected input shape
x_col = x_col.reshape((-1, 64, 64, 4))  # Reshape to (448, 64, 64, 1)
y_col = tf.keras.utils.to_categorical(y_col, num_classes)
# Image preprocessing
datagen = ImageDataGenerator(**preprocess_params)
datagen.fit(x_col)

# Train the model
model.fit(datagen.flow(x_col, y_col, batch_size=32), epochs=2)


image_path = '/Users/saikoushikmupparapu/Desktop/Intern/CT_RETINA/AGE_RMD_55/AR10Test/AMRD49.jpeg'
image = cv2.imread(image_path)
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (64, 64))
image = image.astype(np.float32) / 255.0

# Expand dimensions and reshape
input_image = np.expand_dims(image, axis=0)
input_image = np.expand_dims(input_image, axis=-1)
input_image = np.repeat(input_image, 4, axis=-1)

# Make predictions
predictions = model.predict(input_image)


# Interpret predictions
predicted_class = np.argmax(predictions[0])
disease_type = predicted_class + 1  # Assuming disease types start from 1

print("Predicted Disease Type:", disease_type)