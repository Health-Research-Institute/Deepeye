# test file
import cv2

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

#define data directory 
dataDir =  "C:\\Users\\Temp\\Documents\\RetinaImages\\"

#open and read labels file
with open(dataDir + 'labels.log') as f:
    while True:
        line = f.readline()
        if not line:
            break
        # find all lines with 7
        if line.find('7', 7, 20) != -1:
            print(line.strip())

# To read image from disk, we use
# cv2.imread function, in below method,
img = cv2.imread(dataDir + 'im0001.ppm', cv2.IMREAD_COLOR)
#img = cv2.imread("C:\\git\\opencv\\sources\\samples\\data\\baboon.jpg", cv2.IMREAD_COLOR)
#img = cv2.imread("C:\\Users\\Temp\\Documents\\RetinaImages\\im0001.ppm", cv2.IMREAD_COLOR)
# Creating GUI window to display an image on screen
# first Parameter is windows title (should be in string format)
# Second Parameter is image array
cv2.imshow("image", img)
 
# To hold the window on screen, we use cv2.waitKey method
# Once it detected the close input, it will release the control
# To the next line
# First Parameter is for holding screen for specified milliseconds
# It should be positive integer. If 0 pass an parameter, then it will
# hold the screen until user close it.
cv2.waitKey(0)
 
# It is for removing/deleting created GUI window from screen
# and memory
cv2.destroyAllWindows()