# Internship-HRI
The source consists of a total of 5 files.
-	Training_binaryclassifier.py
-	Training_multiclassifier.py
-	Testset_binaryclassifier.py
-	Testset_multiclassifier.py
-	Test_singleimage_multiclassifier.py

  
Output files: Log files are also made to keep track of the results and for images that are used in training.
- binary_output_logs.txt: This log file has the output of binary classifier training
- output_logs.txt: This log filer has the output of multi-classifier training
- binary_training_images_filenames.txt: This has the list of images that are used for binary classifier training
- training_images_filenames.txt: This has the list of images that are used for multi-classifier training.
I have also uploaded the images used for training and testing.
-Multi classifier:  
Folder Structure:
Training-eyetype-AR, MH,DR,NR,CR
Testing- AR, MH,DR,NR,CR
-Binary classifier:  
Folder Structure:
Training-eyetype-ER, NR
Testing-ER,NR
Note: For training use the folder structure that has been mentioned and make necessary  path changes. The training part of the code saves the model and then the test code uses the saved model please make the path changes accordingly
