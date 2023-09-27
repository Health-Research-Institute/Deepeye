import os
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils import image_utils

class_labels = ['AMD','CSR','DR','MH','NORMAL'] 
imageCoreTest = 'Images/CT_RETINA/TempCNNTest'
modelDir = 'Models/cnnModels'
modelName = 'cnn58IperClass_2023-09-27.h5'
deepeye_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")) # Folder containing Deepeye repo
model_path = os.path.join(deepeye_path, modelDir + '/'+ modelName)

thres_factor = 50

#forming log file
logFname = modelName[0:-3] + ".csv"
logDir = 'Logs/cnnLogs'
logPathFile = os.path.join(deepeye_path, logDir + '/'+ logFname)

# Load the trained transfer learning model
tl_model = load_model(model_path)
# Define the path to the test images directory
test_images_dir = os.path.join(deepeye_path, imageCoreTest)

# Initialize variables for accuracy calculation
total_images = 0
correct_predictions = 0
unknown_diagnosys = 0
SickAsNormal = 0
NormalAsSick = 0

# Initialize dictionaries to keep track of counts and correct/incorrect predictions
category_counts = {label: 0 for label in class_labels}
category_correct = {label: 0 for label in class_labels}
category_incorrect = {label: 0 for label in class_labels}
category_unknown = {label: 0 for label in class_labels}

# Lists to store true and predicted labels
true_labels = []
predicted_labels = []

# Loop through each class label and its images
for label in class_labels:
    label_dir = os.path.join(os.path.join(deepeye_path, imageCoreTest), label)
    image_files = os.listdir(label_dir)
    for image_file in image_files:
        image_path = os.path.join(label_dir, image_file)
        img = image_utils.load_img(image_path, target_size=(256, 256))  # Resize to match model input size
        img_array = image_utils.img_to_array(img)
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
        testValues = np.round(100*predictions)
        if np.max(testValues) < thres_factor:
            unknown_diagnosys += 1
            category_unknown[actual_label] += 1
            print('Unknown Diagnosys: ', testValues, 'Label: ', actual_label)
        else: 
            if predicted_label == actual_label:
                correct_predictions += 1
                category_correct[actual_label] += 1
                #print('Correct Predictions: ', np.round(100*predictions), 'Label: ', actual_label)
            else:
                category_incorrect[actual_label] += 1
                print('Incorrect Predictions: ', np.round(100*predictions), 'Labels cor: ', actual_label, 'pred: ', predicted_label)
                if predicted_label == 'NORMAL':
                    SickAsNormal +=1
                elif actual_label == 'NORMAL':
                    NormalAsSick +=1

        
        total_images += 1
        
        # Store true and predicted labels
        true_labels.append(actual_label)
        predicted_labels.append(predicted_label)

# Calculate accuracy
accuracy = correct_predictions / total_images  * 100
ud = unknown_diagnosys/total_images  * 100

logFile = open(logPathFile, "a")
logFile.write("Threshold factor: " + str(thres_factor))
titleString = "\nCorrect with Confidence: {:.2f}%".format(accuracy)
logFile.write(titleString + "\n")
print(titleString)

tS2 ="Unknown Diagnosys: {:.2f}%".format(ud)
print(tS2)
logFile.write(tS2 + "\n")

tS3 ="Incorrect Diagnosys: {:.2f}%".format(100 - ud - accuracy)
print(tS3)
logFile.write(tS3 + "\n")

tS4 = "Normal as Sick: " + str(NormalAsSick) + "; Sick as Normal: " + str(SickAsNormal)
print(tS4)
logFile.write(tS4 + "\n")

# Print category-wise statistics
print("\nCategory-wise Statistics:")
for label in class_labels:
    total = category_counts[label]
    correct = category_correct[label]
    incorrect = category_incorrect[label]
    ud = category_unknown[label]
    
    failure_rate = incorrect / total * 100 if total > 0 else 0
    
    print("Category:", label)
    print("Total:", total)
    print("Confident Correct:", correct)
    print("Unknown", ud)
    print("Incorrect:", incorrect)
    print("Accuracy: {:.2f}%".format(failure_rate))
    print("----------------------")

    logFile.write("\nCategory: " + str(label))
    logFile.write("\nTotal: " + str(total))
    logFile.write("\nConfident Correct: " + str(correct))
    logFile.write("\nUnknown:" +str(ud))
    logFile.write("\nIncorrect:" +str(incorrect))

    logFile.write("\nFailure Rate: {:.2f}%".format(failure_rate))
    logFile.write("\n----------------------")

# Print classification report
print("\nClassification Report:")
class_report = classification_report(true_labels, predicted_labels, target_names=class_labels)
print(class_report)

logFile.write("\nClassification Report:")
logFile.write(class_report)

# Generate confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=class_labels)
print("\nConfusion Matrix:")
print(conf_matrix)

logFile.write("\nConfusion Matrix:\n")
logFile.write(str(conf_matrix))

logFile.close()
