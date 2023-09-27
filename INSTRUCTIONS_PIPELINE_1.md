# Pipeline 1 Instructions

I. Images Set-up
1.1 Copy Image Files from Google Drive into folder Images (same level as DeepEye)
1.2 Check that inside Images there are 5 core folders:
AMD, CSR, DR, MH and NORMAL
1.3 Each of 5 folders should contain two folders All (with images) and All9L (empty)

II. Creation of 9 layers in All9L folders
2.1 Ensure that the model retina_segmentation_8_layer.hdf5 is in Models/segmentModels
Models is same level as Images 
2.2 Run simple9LCreator.py. Run-time is 2-3 minutes.  

III. Split to Test and Train folders and create logs 
3.1 Create folders TempCNNTrain, TempCNNTest, TempDenseTrain, TempDenseTest inside CT_RETINA
3.2 Ensure that you have directory named Logs on the same level as Images
Ensure that inside Logs there is a directory TrainTestNames  
3.3 In fileSorterCNN_Dense.py set-up nTrain to how many images will be used from each class
Note that 3*nTrain will be used for NORMAL
Suggested values are 25, 35, 45 and 55

IV. Train CNN Model
4.1 Run Training_multiclassifier.py. Recommended amount for # epochs is 30. 
4.2 Be sure to have the coresponding directories for Logs/cnnLogs and
Figures/cnnFigures. Do not pay attention for validation accuracy. Model will be validated next on test set.
4.3 Be sure to have directory to save model into Model/denseModels 

V. Train Dense Net Model
5.1 Run 



