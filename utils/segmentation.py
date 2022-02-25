import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
from tqdm import tqdm



_RESCALE_SLOPE = 1
_RESCALE_INTERCEPT = -1024

IMG_EXTENSIONS = ('png', 'jpg', 'jpeg', 'tif', 'bmp')
HU_WINDOW_WIDTH = 1500
HU_WINDOW_CENTER = -1000


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

def _uint16_hu_to_uint8(data):
    data = data.astype(np.float)*_RESCALE_SLOPE + _RESCALE_INTERCEPT
    return ensure_uint8(data)


def find_contours(binary_image):
    """Helper function for finding contours"""
    return cv2.findContours(binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[0]


def body_contour(binary_image):
    """Helper function to get body contour"""
    contours = find_contours(binary_image)
    areas = [cv2.contourArea(cnt) for cnt in contours]
    body_idx = np.argmax(areas)
    return contours[body_idx]


def auto_body_crop(image, scale=1.0):
    """Roughly crop an image to the body region"""
    # Create initial binary image
    filt_image = cv2.GaussianBlur(image, (5, 5), 0)
    filt_image = np.uint8(filt_image)
    thresh = cv2.threshold(filt_image[filt_image > 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    bin_image = np.uint8(filt_image > thresh)
    erode_kernel = np.ones((7, 7), dtype=np.uint8)
    bin_image = cv2.erode(bin_image, erode_kernel)

    # Find body contour
    body_cont = body_contour(bin_image).squeeze()

    # Get bbox
    xmin = body_cont[:, 0].min()
    xmax = body_cont[:, 0].max() + 1
    ymin = body_cont[:, 1].min()
    ymax = body_cont[:, 1].max() + 1

    # Scale to final bbox
    if scale > 0 and scale != 1.0:
        center = ((xmax + xmin)/2, (ymin + ymax)/2)
        width = scale*(xmax - xmin + 1)
        height = scale*(ymax - ymin + 1)
        xmin = int(center[0] - width/2)
        xmax = int(center[0] + width/2)
        ymin = int(center[1] - height/2)
        ymax = int(center[1] + height/2)

    return image[ymin:ymax, xmin:xmax], (xmin, ymin, xmax, ymax)

def validate(scan):
    shape = np.shape(scan)
    number = np.count_nonzero(scan <= -320)
    min = (shape[0])*(shape[1])*0.60
    # print(number, max)
    return number > min


if __name__=="__main__":
    base_path = '/Users/alicebizzarri/PycharmProjects/COVID-CT/dataset/unife/POS/'
    lista = os.listdir(base_path)
    for path in tqdm(lista):
        try:
            directory = 'dataset/unife/png/' + path[:-7]
            if not os.path.exists(directory):
                os.makedirs(directory)
            volume = nib.load(base_path+path).get_fdata()
            volume = np.transpose(volume)
            i = 0
            for idx, scan in zip(range(len(volume)), volume):
                # is_validate = validate(scan)
                # if is_validate:
                scan = _uint16_hu_to_uint8(scan)
                scan, bbox = auto_body_crop(scan)
                scan = cv2.resize(scan, (256, 256))
                cv2.imwrite(directory+"/"+str(idx)+".png" , scan)
        except:
            print("[ERROR] lettura file", path )
    # scan = _uint16_hu_to_uint8(volume[150,:,:])
    # scan, bbox = auto_body_crop(scan)
    # mask = np.zeros(np.shape(scan))
    # mask[(scan<-320)] = 255
    # plt.imshow(mask)
    # plt.show()
    
