import numpy as np
import matplotlib.pyplot as plt
from mpl_point_clicker import clicker
import cv2 as cv
from typing import Iterable

#This function gets all digital numbers from the dictionary produced by clicker
def get_all_values(d):
    if isinstance(d, dict):
        for v in d.values():
            yield from get_all_values(v)
    elif isinstance(d, Iterable) and not isinstance(d, str): # or list, set, ... only
        for v in d:
            yield from get_all_values(v)
    else:
        yield d 

#define initial data 
# data from where to take images
dataDir =  "../Images/CT_RETINA/NORMAL_206"
ext = '.jpeg'
ImageName = '/NORMAL88' #This is file name also to wtite to

#Read image and define its width
image0 = cv.imread(dataDir + ImageName + ext , 0)
imWidth = image0.shape[1]

#define number of xPoints 
numXpts = 26
xInt = np.linspace(0, imWidth, 26)
ExitArray = np.linspace(0, imWidth, 26)

fig, ax = plt.subplots(constrained_layout=True)
ax.imshow(image0, cmap="gray")
#label number of points for each boundary layer 
# # of markers need to correspond to number of objects (lines) to be labeled
klicker = clicker(ax, ['b1', 'b2', 'b3'], markers=['o', 'o', 'o'])
plt.show()

labeledP = klicker.get_positions()
print(labeledP)

for key in labeledP:
    xyPts = labeledP[key]
    #create list of all pixel values 
    rez=list(get_all_values(xyPts))
    #Interpolation Points
    xp = rez[0::2]
    fp = rez[1::2]
    fInt = np.interp(xInt, xp, fp)
    ExitArray = np.vstack([ExitArray,fInt])

    ax.plot(xInt, fInt, color = 'red', linewidth = 3)

#save to .csv file
np.savetxt(dataDir + ImageName + '.csv', ExitArray, fmt="%8.3f", delimiter=",")
print('Done')

