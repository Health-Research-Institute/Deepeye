import os
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

modelName = 'cnnModel2024-09-18-18-27.h5'
mildThreshold = 0.84

#Filenames and paths
imageDirCoreName = 'Images/CT_RETINA_BinaryResearch'
trainingDir = imageDirCoreName + '/cnnBinary/Training'
testingDir = imageDirCoreName + '/cnnBinary/Testing'

deepeye_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) # Folder containing Deepeye repo
up_deepeye_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")) 

# Where reports shoould go
coreInd = modelName.split('l')[1] 
coreInd = coreInd.split('.')[0] 
repName = 'Ternary' + coreInd + '.csv'
stName = 'Binary' +coreInd + '.csv'
triReportName = os.path.join(up_deepeye_path, 'Logs/cnnLogs/' + repName)
biReportName = os.path.join(up_deepeye_path, 'Logs/cnnLogs/' + stName)

# Load the trained transfer learning model
model_path= os.path.join(up_deepeye_path, 'Models/cnnModels/' + modelName)
tl_model = load_model(model_path)

# Define the path to the test images directory
test_images_dir = os.path.join(up_deepeye_path, testingDir)

# Get the list of class labels (subdirectories)

class_labels = sorted(os.listdir(test_images_dir))
if len(class_labels)==3:
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
probability = []
imageName =[]

# Loop through each class label and its images
for label in class_labels:
    label_dir = os.path.join(os.path.join(up_deepeye_path, testingDir), label)
    image_files = os.listdir(label_dir)
    for image_file in image_files:
        image_path = os.path.join(label_dir, image_file)
        img = image.load_img(image_path, target_size=(256, 256))  # Resize to match model input size
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.  # Normalize
        
        # Predict class probabilities
        predictions = tl_model.predict(img_array)
        prediction = predictions[0]
        
        # Get predicted label index
        predicted_label_idx = np.argmax(prediction)
        
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
        probability.append(prediction)
        imageName.append(image_file)

# Calculate accuracy
accuracy = correct_predictions / total_images * 100


# Print basic statistics
print("\nBasic Statistics:")
print("Test set accuracy: {:.2f}%".format(accuracy))
#print('Category', class_labels[0], class_labels[1])
print('Total', category_counts)
print('Correct', category_correct)
print('Incorrect', category_incorrect)

logFile = open(biReportName , "w")
#logFile.write('Binary Accuracy', accuracy)
logFile.write('Category: ' + ''.join(class_labels))
logFile.write('Total: ' + ''.join(category_counts))
logFile.write('Correct: ' + ''.join(category_correct))
logFile.write('Incorrect: ' + ''.join(category_incorrect))
logFile.close()

# Print classification report
#print("\nClassification Report:")
#print(classification_report(true_labels, predicted_labels, target_names=class_labels))
# Generate confusion matrix
#conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=class_labels)
#print("\nConfusion Matrix:")
#print(conf_matrix)

total_mild_images = 0
total_fp = 0
total_fn = 0
probReport = pd.DataFrame(columns=['Deep-Eye Label', 'Image-Name', 'If Binary Ok', 'Probabilities'])

for i in range (0, total_images):
    if (predicted_labels[i] != true_labels[i]):
        binaryOk = "No"
        #Calculate Deep-eye label
        #MAKE DIAGNOSIS STRICT if Probability is HIGH
        if probability[i][0] >= mildThreshold:
            deLabel = 'ER'
            total_fn +=1
        elif probability[i][1] >= mildThreshold:
            delabel = 'FALSE POS'
            total_fp +=1
        else:
            deLabel = 'MR'
            total_mild_images +=1

        probReport = pd.concat([probReport, pd.DataFrame.from_records([{'Deep-Eye Label': deLabel, 'Image-Name': imageName[i],  'If Binary Ok': binaryOk, 'Probabilities': np.round(probability[i], 2) }])], ignore_index=True)
        probReport.to_csv(triReportName, index=False)
    elif (max(probability[i]) < mildThreshold):
        binaryOk = "Yes"
        #Calculate Deep-eye label 
        deLabel = 'MR'
        total_mild_images +=1
        probReport = pd.concat([probReport, pd.DataFrame.from_records([{'Deep-Eye Label': deLabel, 'Image-Name': imageName[i],  'If Binary Ok': binaryOk, 'Probabilities': np.round(probability[i], 2) }])], ignore_index=True)
        probReport.to_csv(triReportName, index=False)
    
print('Mildly Classified Images', total_mild_images)

