import cv2
import numpy as np
import os
import pandas as pd



_RESCALE_SLOPE = 1
_RESCALE_INTERCEPT = -512

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
    cv2.imwrite('rova.png', filt_image)
    filt_image = np.uint8(filt_image)
    thresh = cv2.threshold(filt_image[filt_image > 0], 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
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

def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    # volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume



if __name__=="__main__":
    csv = pd.read_csv("dataset/metadata.csv")
    csv = np.array(csv[csv["source"]=="iCTCF"]["patient id"])
    csv = [name.replace("HUST-", "") for name in csv]
    print(csv)
    for i in range(1132, 1522):
        name = f"Patient%20{i}.zip"
        print(name)
        if f"Patient{i}" in csv:
            continue
        elif not os.path.exists(f"dataset/ICTCF/{name}"):
            print("download paziente: ", i)
            os.system(f"/opt/homebrew/bin/wget http://ictcf.biocuckoo.cn/patient/CT/{name} --directory-prefix=dataset/ICTCF/{name}") 

    
#http://ictcf.biocuckoo.cn/patient/CT/Patient%201260.zip