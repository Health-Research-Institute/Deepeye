# Import libraries
import numpy as np
import glob
import PIL

#convert .csv file into 2-D array
def file2matrix(csv_files, indFile):
    #init reading the first file 
    r = np.genfromtxt(csv_files[indFile], delimiter=',', names=True)
    nLevels = len(r)
    nElements = len(r[0])

    #initiate first line 
    ptsMatrix = np.reshape(list(r[0]),(1,nElements))
    ind = 1
    while ind < nLevels:
        ptsMatrix = np.append(ptsMatrix, np.reshape(list(r[ind]),(1,nElements)), axis=0)
        ind = ind + 1
    return ptsMatrix    


#define data folder to read/write .csv files
dataDir =  "../Images/CT_RETINA/NORMAL_206"
# Get CSV files list from a folder
csv_files = glob.glob(dataDir + "/*.csv")
nFiles = len(csv_files)

#initialise mean colection matrix
meanMatrix = file2matrix(csv_files, 0)

#Loop to add up
ind2 = 1
while ind2 < nFiles: 
      meanMatrix = meanMatrix + file2matrix(csv_files, ind2)
      ind2 = ind2 + 1

#difide by number of elements to get mean
meanMatrix = meanMatrix/nFiles

#initialise std colection matrix
stdMatrix = np.square(file2matrix(csv_files, 0) - meanMatrix)

#Loop to add up
ind3 = 1
while ind3 < nFiles: 
      stdMatrix = stdMatrix + np.square(file2matrix(csv_files, ind3) - meanMatrix)
      ind3 = ind3 + 1

#calculate final standard deviation matrix
stdMatrix = np.sqrt(stdMatrix/nFiles)

#write mean and standard deiations into new files 
np.savetxt(dataDir + "/normalMean.csv" , meanMatrix, fmt="%8.3f", delimiter=",")
np.savetxt(dataDir + "/normalSTD.csv"  , stdMatrix, fmt="%8.3f", delimiter=",")

#create now image of meanMatrix

im = PIL.Image.new(mode="RGB", size=(200, 200))





print('Done')

