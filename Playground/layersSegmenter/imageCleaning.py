# Image Cleaning

def eraseObject(imageIn,rangeI, rangeJ):
    imageOut = imageIn.copy()
    i=rangeI[0]
    j=rangeJ[0]
    while  i<= rangeI[1]:
        while j <= rangeJ[1]:
            imageOut[j, i] = imageOut[j, i+120]
            #imageOut[j, i] = 0
            j=j+1
        i=i+1
    return imageOut
        








