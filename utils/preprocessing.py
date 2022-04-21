from enum import auto
import cv2
import matplotlib.pyplot as plt
import nibabel
import numpy as np
import os
import pandas as pd
import numpy.random as ran
from tqdm import tqdm

IMG_EXTENSIONS = ('png', 'jpg', 'jpeg', 'tif', 'bmp')
HU_WINDOW_WIDTH = 1500
HU_WINDOW_CENTER = -600

def moment( n, bins, pdf ):
        if ( n < 2 ):
            print( "Errore! il primo parametro DEVE essere maggiore o uguale a 2!" )
            return None
        else:
            nextr = len(bins)
            x = 0.5 * ( bins[0:nextr-1] + bins[1:nextr] )
            dx = x[1] - x[0]
            f_ave = x * pdf
            xm = dx * f_ave.sum()
            f_ave = ( ( x - xm )**n ) * pdf
            return dx * f_ave.sum()

def hu_to_uint8(hu_images, window_width, window_center):
    """Converts HU images to uint8 images"""
    images = (hu_images.astype(np.float) - window_center + window_width/2)/window_width
    uint8_images = np.uint8(255.0*np.clip(images, 0.0, 1.0))
    return uint8_images


def ensure_uint8(data, window_width=HU_WINDOW_WIDTH, window_center=HU_WINDOW_CENTER):
    """Converts non-uint8 data to uint8 and applies window level to HU data"""
    if data.dtype != np.uint8:
        if data.ptp() > 255:
            # Assume HU
            data = hu_to_uint8(data, window_width, window_center)
        else:
            # Assume uint8 range with incorrect dtype
            data = data.astype(np.uint8)
    return data

def find_contours(binary_image):
    """Helper function for finding contours"""
    return cv2.findContours(binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[0]


def body_contour(binary_image):
    """Helper function to get body contour"""
    contours = find_contours(binary_image)
    areas = [cv2.contourArea(cnt) for cnt in contours]
    sorteddata = sorted(zip(areas, contours), key=lambda x: x[0], reverse=True)
    # body_idx = np.argmax(areas)
    # print(np.shape(sorteddata[0][1]), np.shape(contours[body_idx]))
    
    return sorteddata
    # return contours[body_idx]



def auto_body_crop(image, scale=1.0):
    """Roughly crop an image to the body region"""
    # Create initial binary image
    filt_image = cv2.GaussianBlur(image, (5, 5), 0)
    thresh = cv2.threshold(filt_image[filt_image > 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    bin_image = np.uint8(filt_image > thresh)
    erode_kernel = np.ones((7, 7), dtype=np.uint8)
    bin_image = cv2.erode(bin_image, erode_kernel)
    # cv2.imwrite("bin.png", bin_image*255)

    # Find body contour
    sorted_contourn = body_contour(bin_image)
    # Get bbox
    try:
        lung1, lung2 = sorted_contourn[1][1],  sorted_contourn[2][1]
    except:
        return None, False
    xmin1, xmax1, ymin1, ymax1 = lung1[:, 0, 0].min(),lung1[:, 0, 0].max(),lung1[:, 0, 1].min(),lung1[:, 0, 1].max()
    xmin2, xmax2, ymin2, ymax2 = lung2[:, 0, 0].min(),lung2[:, 0, 0].max(),lung2[:, 0, 1].min(),lung2[:, 0, 1].max()
    xmin = min(xmin1, xmin2)
    xmax = max(xmax1, xmax2) + 1
    ymin = min(ymin1, ymin2)
    ymax = max(ymax1, ymax2) + 1

    # Scale to final bbox
    if scale > 0 and scale != 1.0:
        center = ((xmax + xmin)/2, (ymin + ymax)/2)
        width = scale*(xmax - xmin + 1)
        height = scale*(ymax - ymin + 1)
        xmin = int(center[0] - width/2)
        xmax = int(center[0] + width/2)
        ymin = int(center[1] - height/2)
        ymax = int(center[1] + height/2)
    
    minimo = int(np.shape(image)[1]*0.10)
    nx = int(np.shape(image)[0]*0.40)  
    ny = int(np.shape(image)[1]*0.40)        
    #draw it
    # controllo bordi
    top = xmin
    buttom = np.shape(image)[0]-xmax
    left = ymin
    right = np.shape(image)[1]-ymax
    if  top > minimo and buttom > minimo and left > minimo and right > minimo:
        border = True
    else:
        border = False
    if xmax-xmin>nx and ymax-ymin>ny and border:
        validate = True
    else:
        validate = False
        
    return image[ymin:ymax, xmin:xmax], validate

if __name__=="__main__":
    base_path = 'unife/'
    pazienti = os.listdir(base_path)
    dir = 'dataset/unife/preproc/'
    if not os.path.exists(dir):
        os.system("mkdir " + dir)
    for p in pazienti:
        nib_p = nibabel.load(base_path + p).get_data()
        nib_p = nib_p.transpose(2,1,0)
        for i, scan in enumerate(nib_p):
            scan = hu_to_uint8(scan, HU_WINDOW_WIDTH, HU_WINDOW_CENTER)
            scan, validate = auto_body_crop(scan)
            if validate:
                cv2.imwrite(f"{dir}{p[:-7]}_{i}.png", scan)
        
            