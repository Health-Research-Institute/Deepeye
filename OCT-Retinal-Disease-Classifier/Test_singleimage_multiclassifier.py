import numpy as np
from keras.preprocessing import image
from keras.models import load_model

# Load the trained transfer learning model
model_path = '/Users/saikoushikmupparapu/Desktop/Intern/best_tl_model_1.h5'
tl_model = load_model(model_path)

# Load and preprocess a single input image for prediction
input_image_path = '/Users/saikoushikmupparapu/Desktop/Intern/Testing/AR/AMRD46.jpeg'
input_image = image.load_img(input_image_path, target_size=(256, 256))  # Resize to match model input size
input_image_array = image.img_to_array(input_image)
input_image_array = np.expand_dims(input_image_array, axis=0)
input_image_array /= 255.  # Normalize

# Predict class probabilities
predictions = tl_model.predict(input_image_array)

# Get the predicted label index
predicted_label_idx = np.argmax(predictions[0])

# Get the list of class labels (subdirectories)
class_labels = ['AR', 'CR', 'DR', 'MH', 'NR']

# Get the predicted label
predicted_label = class_labels[predicted_label_idx]

# Print the predicted label
print("Predicted Label:", predicted_label)
