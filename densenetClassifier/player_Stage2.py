import os
#import glob
#import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim as optim
#import torchvision
import os.path
from helperFunctions import tile
from analysisFunctions import analyseTest
from dataClasses import DataPreprocessingPlayer
from dataClasses import DataPreprocessing
from dataClasses import DenseNet121
import dotenv
dotenv.load_dotenv()

# Check if PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")

# Data Preprocessing 
data = DataPreprocessing()

train_set, test_set = random_split(data, [0, 1])
testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

if device == "mps":
    model = DenseNet121().to(device)
    model = nn.DataParallel(model).to(device)
elif device == "cuda":
    model = DenseNet121().cuda()
    model = nn.DataParallel(model).cuda()
else:
    model = DenseNet121()
    model = nn.DataParallel(model)

#load model
model = torch.load(os.getenv('goldenModel'))

#forming log file
logFname = "log" + str(os.getenv('nClasses')) + "class" + str(os.getenv('downImageSize')) + ".csv"
logPathFile = os.getenv('player_load_MP') + logFname
logFile = open(logPathFile, "a")
titleString = "ProblemType ImageName C# FitScore "
logFile.write(titleString + "\n")

#model evaluation
# model.eval()

# Init statistics
perfect8 = 0
sevenLev = 0
problem = 0

#main loop
useImLevels = eval(os.getenv("useImLevels"))
with torch.no_grad():
    for i, (images, labels, image_names) in enumerate(testloader, 0):
        if device == "mps":
            images = images.to(device)
            labels = tile(labels, 0, useImLevels).to(device)
        elif device == "cuda":
            images = images.cuda()
            labels = tile(labels, 0, useImLevels).cuda() 
        else:
            labels = tile(labels, 0, useImLevels)
        n_batches, n_crops, channels, height, width = images.size()
        image_batch = torch.autograd.Variable(images.view(-1, channels, height, width))
        outputs = model(image_batch)

        # Calculate 
        fitN = analyseTest(outputs)
        
        labeled_as = np.argmax(labels[0].cpu().numpy())
        logString =  image_names[0][0].split("_1_")[0] + " " + str(labeled_as+1) + " " + str(fitN.astype(int))

        # Print all images
        print("#", i+1,"Name:", image_names[0][0].split("_1_")[0], "Class #:", labeled_as+1, "Prob fit:", fitN.astype(int) )

        # Log only problematic Images:
        if max(fitN) == 8:
            perfect8 = perfect8 + 1
        elif max(fitN) == 7:
            sevenLev = sevenLev + 1
        else: # Max fit < 7
            problem = problem + 1

        #if (labeled_as == nCl-1) and (fitN[nCl-1]< 7):
            logFile.write('Doctor to Look' + logString + "\n")

logFile.close()

print('Total images tested: ', len(testloader))
print('Images with all 8s: ', perfect8, ' with 7s: ', sevenLev) 
print('Accurate classification with 99.8% Conf,', 100*perfect8/len(testloader))
print('Accurate classification with at least 90% Conf,', 100*(perfect8+sevenLev)/len(testloader))
