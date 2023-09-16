# Analysis Functions
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def show_image_list(list_images, list_titles=None, list_cmaps=None, grid=True, num_cols=2, figsize=(20, 10), \
    title_fontsize=30):
    '''
    Shows a grid of images, where each image is a Numpy array. The images can be either
    RGB or grayscale.

    Parameters:
    ----------
    images: list
        List of the images to be displayed.
    list_titles: list or None
        Optional list of titles to be shown for each image.
    list_cmaps: list or None
        Optional list of cmap values for each image. If None, then cmap will be
        automatically inferred.
    grid: boolean
        If True, show a grid over each image
    num_cols: int
        Number of columns to show.
    figsize: tuple of width, height
        Value to be passed to pyplot.figure()
    title_fontsize: int
        Value to be passed to set_title().
    '''

    assert isinstance(list_images, list)
    assert len(list_images) > 0
    assert isinstance(list_images[0], np.ndarray)

    if list_titles is not None:
        assert isinstance(list_titles, list)
        assert len(list_images) == len(list_titles), '%d imgs != %d titles' % (len(list_images), len(list_titles))

    if list_cmaps is not None:
        assert isinstance(list_cmaps, list)
        assert len(list_images) == len(list_cmaps), '%d imgs != %d cmaps' % (len(list_images), len(list_cmaps))

    num_images  = len(list_images)
    num_cols    = min(num_images, num_cols)
    num_rows    = int(num_images / num_cols) + (1 if num_images % num_cols != 0 else 0)

    # Create a grid of subplots.
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    
    # Create list of axes for easy iteration.
    if isinstance(axes, np.ndarray):
        list_axes = list(axes.flat)
    else:
        list_axes = [axes]

    for i in range(num_images):

        img    = list_images[i]
        title  = list_titles[i] if list_titles is not None else 'Image %d' % (i)
        cmap   = list_cmaps[i] if list_cmaps is not None else (None if img_is_color(img) else 'gray')
        
        list_axes[i].imshow(img, cmap=cmap)
        list_axes[i].set_title(title, fontsize=title_fontsize) 
        list_axes[i].grid(grid)

    for i in range(num_images, len(list_axes)):
        list_axes[i].set_visible(False)

    fig.tight_layout()
    _ = plt.show()

    fig.savefig('plot.png')


def img_is_color(img):

    if len(img.shape) == 3:
        # Check the color channels to see if they're all the same.
        c1, c2, c3 = img[:, : , 0], img[:, :, 1], img[:, :, 2]
        if (c1 == c2).all() and (c2 == c3).all():
            return True

    return False


def insert_patch_subpixel2(im, patch, p):
    """
    im: numpy array with source image.
    patch: numpy array with patch to be inserted into the source image
    p: tuple with the center of the position (can be float) where the patch is to be inserted.
    """
    ths = patch.shape[0]/2
    xpmin = p[0] - ths
    ypmin = p[1] - ths
    Ho = np.array([[1, 0, xpmin],
                   [0, 1, ypmin],
                   [0, 0,     1]], dtype=float)

    h,w = im.shape
    im2 = cv.warpPerspective(patch, Ho, (w,h),
                        flags=cv.INTER_LINEAR,
                        borderMode=cv.BORDER_CONSTANT)

    patch_mask = np.ones_like(patch,dtype=float)
    blend_mask = cv.warpPerspective(patch_mask, Ho, (w,h),
                        flags=cv.INTER_LINEAR,
                        borderMode=cv.BORDER_CONSTANT)

    #I don't multiply im2 by blend_mask because im2 has already
    #been interpolated with a zero background.
    im3 = im*(1-blend_mask)+im2
    im4 = cv.convertScaleAbs(im3)
    return im4
