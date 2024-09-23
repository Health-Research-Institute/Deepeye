Binary Research Package.
This package contains the following files/functions: 
splitNamesTrainTest.py
copyTrainTestToDirs.py
cnnTrain.py
denseTrain.py - does not exist yet. To be created based on existing Dense Net Classifier 
cnnTest.py
denseTest.py - does not exist yet. To be created based on existing Dense Net Classifier 
ternaryAnalyser.py - currently implemented at the end of cnnTest.py, based on CNN only results
Order of operations and options: 
1. splitNamesTrainTest.py
As input assumes Image directories and names to be split into Train and Test sets. Note that any additional images can be added to Test set before cnnTest.py and denseTest.py
Example input:
classNames =[‘AMD’,’CSR’,’DR,’MH’,’NORMAL’]
nTrain = [25,25,30,16,75]
As a result this function produces two files in Logs/TrainTestNames
with the following structure Names_date-time train.csv and Names_date-time test.csv and. 
For example Names_2024-09-16-16-29train.csv file has been created on September 16, 2024 at 16:29
2. copyTrainTestToDirs.py
This function first erases everything inside CT_RETINA_BinaryResearch, and then creates folders structure with two folders cnnBinary and denseBinary. Inside each of these directories there are Training and Testings subdirectories. Inside these two there are ER and NR directories for Sick and Normal images respectively.
As input it receives the Names_date-time train.csv file. 
It copies files accordingly and also creates dataIndex.txt file based on the date-time that is inherited from Names_date-time train.csv file. 
After creation of temporary directories other files could be added to Testing directories 
NOTE: If Software Engineering allows we can potentially eliminate this file operating subsequent functions from the lists created by splitNamesTrainTest.py
3. cnnTrain.py
Takes files from CT_RETINA_BinaryResearch/cnnBinary/Training  and dateIndex.txt and creates CNN Model, which is written into Models/cnnModels. CNN model is written with the same inherited date, for example cnnModel2024-09-16-16-29.h5
Recommended: 50 epochs
4. denseTrain.py
Takes files from CT_RETINA_BinaryResearch/denseBinary/Training  and dateIndex.txt and creates Dense Net Model, which is written into Models/denseModels. CNN model is written with the same inherited date, for example denseModel2024-09-16-16-29.h5
Recommended: ?? epochs
5. cnnTest.py
Creates a table of File Name, Original Label (ER or NR) and 2-numbers probability vector. Note that sum of two probabilities in this method is not necessary equal to 1
6. denseTest.py
Creates a table of File Name, Original Label (ER or NR) and 16-numbers probability tensor.
7. ternaryAnalyser.py - currently implemented inside cnnTest.py
Analyses Labels and Probabililites and outputs the classification results and statistics 
Curently results are written into Logs/cnnLogs as BinaryDate-Time.csv and TernaryDate-Time.csv. Location should be redefined when denseNet is implemented
BinaryDate-Time contains the results of brute-force binary classification 
TernaryAnalyser uses thresholds (we found the value of 0.85 is the best for CNN) to separate images into 3 classes: ER - Sick, NR - Normal and MR - mildly sick