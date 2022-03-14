import cv2
import nibabel as nib
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

IMG_EXTENSIONS = ('png', 'jpg', 'jpeg', 'tif', 'bmp')
HU_WINDOW_WIDTH = 1500
HU_WINDOW_CENTER = -600

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
    cv2.imwrite("bin.png", bin_image*255)

    # Find body contour
    sorted_contourn = body_contour(bin_image)
    # Get bbox
    lung1, lung2 = sorted_contourn[1][1],  sorted_contourn[2][1]
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
        
    #draw it

    return image[ymin:ymax, xmin:xmax], (xmin, ymin, xmax, ymax)


# base_path = "../ictcf.biocuckoo.cn/patient/"
base_path = "dataset/"
csv = pd.read_csv("dati_clinici_superfinal.csv")
ids = np.array(csv["id"])
covids = np.array(csv['covid'])
CTs = np.array(csv['CT'])
directory = base_path + 'prova'
print("[INFO]", directory)
if not os.path.exists(directory):
    os.makedirs(directory)
normal = len(csv[csv["CT"]=="Negative"])
common = len(csv[csv["CT"]=="Positive"][csv["covid"]=="Negative"])
covid = len(csv[csv["CT"]=="Positive"][csv["covid"]=="Positive"])
n_classi = [int(normal*0.8), int(common*0.8), int(covid*0.8)]
print(n_classi)
#n_classi = [1,1,1]
#ids = ['patient_46', 'patient_47']
#covids = ['Negative', 'Negative']
#CTs = ['Positive', 'Positive']
for name, covid, ct in zip(ids, covids, CTs):
    if ct == "Positive" and covid == "Positive":
        label = "2_covid"
        n = n_classi[2]
        n_classi[2]-=1
    elif ct == "Positive" and covid =="Negative":
        label = "1_common"
        n = n_classi[1]
        n_classi[1]-=1
    elif ct == "Negative":
        label="0_normal"
        n = n_classi[0]
        n_classi[0]-=1
    else:
        continue
    name = name.replace(" ", "_")
    name = name.replace("P", "p") + ".nii.gz"
    print("[INFO]", name)
    os.system(f"python utils/nii2png.py -i {base_path}nifti/{name} -o {directory}/{label}/{name[:-7]}")
    imgs = os.listdir(f"{directory}/{label}/{name[:-7]}/")
    for idx, p in enumerate(imgs):
        image = cv2.imread(f"{directory}/{label}/{name[:-7]}/{p}", 0)
        #try:
        image, _ = auto_body_crop(image)
        cv2.imwrite(f"{directory}/{label}/{name[:-7]}_{idx}.png", image)
        #except:
        #    print("ERRORE:", name[:-7])
