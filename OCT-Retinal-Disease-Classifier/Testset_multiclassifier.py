import os
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

deepeye_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")) # Folder containing Deepeye repo

# Load the trained transfer learning model
model_path = os.path.join(deepeye_path, 'Models/SaiModels/best_tl_multi_model.h5')
tl_model = load_model(model_path)

# Define the path to the test images directory
test_images_dir = os.path.join(deepeye_path, 'Images/SaiImages/MultiClassifier/Testing')

# Get the list of class labels (subdirectories)
class_labels = sorted(os.listdir(test_images_dir))
class_labels.pop(0)

# Initialize variables for accuracy calculation
total_images = 0
correct_predictions = 0

# Initialize dictionaries to keep track of counts and correct/incorrect predictions
category_counts = {label: 0 for label in class_labels}
category_correct = {label: 0 for label in class_labels}
category_incorrect = {label: 0 for label in class_labels}

# Lists to store true and predicted labels
true_labels = []
predicted_labels = []

# Loop through each class label and its images
for label in class_labels:
    label_dir = os.path.join(os.path.join(deepeye_path, 'Images/SaiImages/MultiClassifier/Testing'), label)
    image_files = os.listdir(label_dir)
    for image_file in image_files:
        image_path = os.path.join(label_dir, image_file)
        img = image.load_img(image_path, target_size=(256, 256))  # Resize to match model input size
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.  # Normalize
        
        # Predict class probabilities
        predictions = tl_model.predict(img_array)
        
        # Get predicted label index
        predicted_label_idx = np.argmax(predictions[0])
        
        # Get true label index
        true_label_idx = class_labels.index(label)
        
        # Get the actual and predicted labels
        actual_label = label
        predicted_label = class_labels[predicted_label_idx]
        
        # Update category counts
        category_counts[actual_label] += 1
        
        # Check if prediction is correct
        if predicted_label == actual_label:
            correct_predictions += 1
            category_correct[actual_label] += 1
        else:
            category_incorrect[actual_label] += 1
        
        total_images += 1
        
        # Store true and predicted labels
        true_labels.append(actual_label)
        predicted_labels.append(predicted_label)

# Calculate accuracy
accuracy = correct_predictions / total_images * 100
print("Test set accuracy: {:.2f}%".format(accuracy))

# Print category-wise statistics
print("\nCategory-wise Statistics:")
for label in class_labels:
    total = category_counts[label]
    correct = category_correct[label]
    incorrect = category_incorrect[label]
    accuracy = correct / total * 100 if total > 0 else 0
    
    print("Category:", label)
    print("Total:", total)
    print("Correct:", correct)
    print("Incorrect:", incorrect)
    print("Accuracy: {:.2f}%".format(accuracy))
    print("----------------------")

# Print classification report
print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels, target_names=class_labels))

# Generate confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=class_labels)
print("\nConfusion Matrix:")
print(conf_matrix)
