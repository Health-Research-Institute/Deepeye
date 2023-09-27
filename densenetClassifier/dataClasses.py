# Import libraries
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
import torch.nn as nn
import os
import glob
import cv2
from helperFunctions import create_classEncoding
from helperFunctions import create_image_stack
import dotenv
dotenv.load_dotenv()

class DataPreprocessing(Dataset):

    trainMode = eval(os.getenv("trainMode"))
    if trainMode: 
        classPaths = {
        'AMD': ['../../Images/CT_RETINA/TempDenseTrain9L/AMD', 25],
        'CSR':  ['../../Images/CT_RETINA/TempDenseTrain9L/CSR/', 25],
        'DR':   [ '../../Images/CT_RETINA/TempDenseTrain9L/DR/', 25],
        'MH':   ['../../Images/CT_RETINA/TempDenseTrain9L/MH/', 25],
        'NORMAL': ['../../Images/CT_RETINA/TempDenseTrain9L/NORMAL/', 75]
        }
    else: 
        classPaths = {
        'AMD': ['../../Images/CT_RETINA/TempDenseTest9L/AMD', 72-25],
        'CSR':  ['../../Images/CT_RETINA/TempDenseTest9L/CSR/', 63-25],
        'DR':   [ '../../Images/CT_RETINA/TempDenseTest9L/DR/', 108-25],
        'MH':   ['../../Images/CT_RETINA/TempDenseTest9L/MH/', 82-25],
        'NORMAL': ['../../Images/CT_RETINA/TempDenseTest9L/NORMAL/', 187-75]
        }
            
    
    def __init__(self, classPaths=classPaths):
        
        self.image_paths = []
        self.labels = []
        self.images = []

        self.classes = eval(os.getenv("CLASSES"))
        self.classEncoding = create_classEncoding(self.classes)
        self.nImLevelsData = eval(os.getenv("nImLevelsData"))
        self.downImageSize = eval(os.getenv("downImageSize"))
    
        # image paths
        for imCoreName in (self.classEncoding.keys()):
            temp_paths = []
            for directoryPath in glob.glob(classPaths[imCoreName][0]):
                for imgPath in glob.glob(os.path.join(directoryPath, "*.jpg")):
                    temp_paths.append(imgPath)

            # labels 
            labels_list = [imCoreName] * classPaths[imCoreName][1] * self.nImLevelsData

            for label in labels_list:
                labelTensor = torch.FloatTensor(np.zeros(len(self.classes)))

                labelTensor = labelTensor.add(self.classEncoding[label])
                self.labels.append(labelTensor)

            img_paths = temp_paths[:classPaths[imCoreName][1] * self.nImLevelsData] 
            self.image_paths.append(img_paths)

            for image_path in img_paths:
                img = cv2.imread(image_path,0) 
                img = cv2.resize(img, (self.downImageSize, self.downImageSize), interpolation = cv2.INTER_AREA)
                img = np.reshape(img, (*img.shape, 1))
                img = np.transpose(img, (2, 0, 1))
                self.images.append(img)
              
        self.image_paths = [y for x in self.image_paths for y in x]
        self.stacked_images, self.stacked_labels, self.stacked_image_names  = create_image_stack(self.images, self.labels, self.image_paths, self.nImLevelsData)  
                
    def __getitem__(self, index):
        # preprocess and return single image stack of dim 8*1*224*224
        return self.stacked_images[index], self.stacked_labels[index], self.stacked_image_names[index]
    
    def __len__(self):

        return len(self.stacked_images)
    

class DenseNet121(nn.Module):
    def __init__(self):
        super(DenseNet121, self).__init__()

        self.classes = eval(os.getenv("CLASSES"))
        self.model = torchvision.models.densenet121(pretrained = True)

        num_ftrs = self.model.classifier.in_features
        self.model.features[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, len(self.classes)),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.model(x)
        return x
    