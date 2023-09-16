# Retina Analysis
import cv2 as cv
#import matplotlib.pyplot as plt
from analysisFunctions import show_image_list
from analysisFunctions import insert_patch_subpixel2

#define data directory 
dataDir =  "../Images/CT_RETINA/NORMAL_206"

image0 = cv.imread(dataDir + '/NORMAL90.jpeg', 0)
#image0 = Image.open(dataDir + '/NORMAL90.jpeg')
patch = image0[380:499, 120:120+120].copy()
#image1 = eraseObject(image0, [0, 119], [380, 499])

#r,c = np.where(image1<50) 
#image1[(r,c)]=255
image1 = insert_patch_subpixel2(image0, patch, [60, 440])
        
image2 = cv.Sobel(image1, cv.CV_64F, 0,1)
#absSobelx = np.absolute(image2) # Absolute x derivateive to accentuate lines
#image3 = np.uint8(255*absSobelx/np.max(absSobelx))

flg, image3 = cv.threshold(image2, 170, 255, cv.THRESH_BINARY)

flg, image4 = cv.threshold(image2, 110, 255, cv.THRESH_BINARY)

#image4 = cv.medianBlur(image3, 3)


list_images = [image0, image1, image2, image3, image4]

show_image_list(list_images, grid=False, num_cols=3, figsize=(10, 10))









