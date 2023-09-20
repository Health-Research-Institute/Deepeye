import os
import numpy as np
import shutil
from torch.utils.data import random_split

classNames = ['AMD','CSR','DR','MH','NORMAL']
nClasses = len(classNames)
nTrain = 40

for testClass in range(0,nClasses):
    imagesPathRead = '../Images/CT_RETINA/' + classNames[testClass] + '/All'
    imagesPathWriteTest = '../Images/CT_RETINA/' + classNames[testClass] + '/Test'
    imagesPathWriteTrain = '../Images/CT_RETINA/' + classNames[testClass] + '/Train'

    file_names = os.listdir(imagesPathRead)
    nFiles = len(file_names)
    indArray = np.arange(0, nFiles)
    if (testClass == nClasses -1): #normal class to have more images
        nTrain = 3*nTrain
    train_set, test_set = random_split(indArray, [nTrain, nFiles-nTrain])
   
    trainInd = np.sort(train_set)
    for i in range(0, len(trainInd)):
        file_name = file_names[trainInd[i]] 
        shutil.move(os.path.join(imagesPathRead, file_name), imagesPathWriteTrain)

print('Moving Files from ALL to Train is done')   