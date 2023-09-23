import os
import numpy as np
import shutil
from torch.utils.data import random_split

classNames = ['AMD','CSR','DR','MH','NORMAL']
nClasses = len(classNames)
nTrain = 40

for splitClass in range(0,nClasses):
    imagesPathRead = '../Images/CT_RETINA/' + classNames[splitClass] + '/All'
    imagesPathWriteTest = '../Images/CT_RETINA/CNNTest/' + classNames[splitClass]
    imagesPathWriteTrain = '../Images/CT_RETINA/CNNTrain/' + classNames[splitClass]

    file_names = os.listdir(imagesPathRead)
    nFiles = len(file_names)
    indArray = np.arange(0, nFiles)
    if (classNames[splitClass] == 'NORMAL'): #normal class to have more images
        train_set, test_set = random_split(indArray, [3*nTrain, nFiles-3*nTrain])
    else:
        train_set, test_set = random_split(indArray, [nTrain, nFiles-nTrain])
   
    trainInd = np.sort(train_set)
   
    for i in range(0, len(trainInd)):
        file_name = file_names[trainInd[i]] 
        shutil.copy(os.path.join(imagesPathRead, file_name), imagesPathWriteTrain)

    testInd = np.sort(test_set)

    for j in range(0, len(testInd)):
        file_name = file_names[testInd[j]] 
        shutil.copy(os.path.join(imagesPathRead, file_name), imagesPathWriteTest)

print('Moving Files from ALL to Train is done')   