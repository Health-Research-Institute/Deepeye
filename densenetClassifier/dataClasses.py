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
    
    # classPaths --> className: [path, #training_samples]

    testOnly = eval(os.getenv("testOnly"))
    if testOnly == 'False': #training regime
        subdir = 'Train'
        imagesFromEachClass = 18
    else:
        subdir = 'Test'
        imagesFromEachClass = 106

    classPaths = {
    
    'AGE_RMD': ['../../Images/CT_RETINA/AGE_RMD_55/AR' + subdir + '9L', min(imagesFromEachClass, 27)],
    'CSR':     ['../../Images/CT_RETINA/CSR_102/CR' + subdir + '9L', min(imagesFromEachClass,62)],
    'DIABETR': [ '../../Images/CT_RETINA/DIABETR_107/DR' + subdir + '9L', min(imagesFromEachClass,65)],
    'MACHOLE': ['../../Images/CT_RETINA/MACHOLE_102/MH' + subdir + '9L', min(imagesFromEachClass,62)],
    'NORMAL':  ['../../Images/CT_RETINA/NORMAL_206/NR' + subdir + '9L', min(3*imagesFromEachClass,106)] 
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