import cv2
import matplotlib.pyplot as plt
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
    
    minimo = int(np.shape(image)[1]*0.10)
    nx = int(np.shape(image)[0]*0.30)  
    ny = int(np.shape(image)[1]*0.30)        
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
    ## LEGGO IMMAGINE TEST E IMMAGINE TRAINING RANDOM:
    test_df = pd.read_csv('total_test_data.csv')
    test_df = test_df['filename']
    train_df = pd.read_csv('total_train_data.csv')
    train_df = train_df['filename']
    test_name = np.array(test_df.sample(n=100))
    train_name = np.array(train_df.sample(n=100))
    test_img = []
    train_img = []
    for name in test_name:
        # print(name)
        if os.path.exists(name):
            test_img.append(cv2.imread(name) / 255.)
    for name in train_name:
        if os.path.exists(name):
            train_img.append(cv2.imread(name) / 255.)
    # print(train_name, '\n', test_name)
    Nc = 100
    pdf_tr = []
    bins_tr = []
    for img in train_img:
        pdf, bins = np.histogram(img, Nc, density = True )
        nextr = len( bins )
        pdf_tr.append(pdf)
        bins_tr.append(bins)
        
    pdf_ts = []
    bins_ts = []
    for img in test_img:
        pdf, bins = np.histogram(img, Nc, density = True )
        nextr = len( bins )
        pdf_ts.append(pdf)
        bins_ts.append(bins)

    # fig, axs = plt.subplots(2)
    # axs[0].bar( bins_tr[:nextr_tr-1], pdf_tr, dx_tr, color = "g" )
    # axs[0].set_title('train')
    # axs[1].bar( bins[:nextr-1], pdf, dx, color = "g" )
    # axs[1].set_title('test')
    
    plt.show()

    
    moment2_tr = np.mean([moment(2, bins, pdf) for bins, pdf in zip(bins_tr, pdf_tr)])
    moment2_ts = np.mean([moment(2, bins, pdf) for bins, pdf in zip(bins_ts, pdf_ts)])
    moment3_tr = np.mean([moment(3, bins, pdf) for bins, pdf in zip(bins_tr, pdf_tr)])
    moment3_ts = np.mean([moment(3, bins, pdf) for bins, pdf in zip(bins_ts, pdf_ts)])
    moment4_tr = np.mean([moment(4, bins, pdf) for bins, pdf in zip(bins_tr, pdf_tr)])
    moment4_ts = np.mean([moment(4, bins, pdf) for bins, pdf in zip(bins_ts, pdf_ts)])
    
    # Calcola i centri dei bins
    xc = 0.5 * ( bins[0:nextr-1] + bins[1:nextr] )
    print( "TEST - Deviazione standard: ", moment2_ts )
    print( "TRAIN - Deviazione standard: ", moment2_tr) 
    print( "TEST - Skewness: ", moment3_ts )
    print( "TRAIN - Skewness: ", moment3_tr)
    print( "TEST - Flatness: ", moment4_ts )
    print( "TRAIN - Flatness: ", moment4_tr )

    
    # base_path = "../ictcf.biocuckoo.cn/patient/"
    # base_path = "dataset/"
    # csv = pd.read_csv("dataset/train_COVIDx_CT-2A.txt", sep=' ')
    # csv = np.array(csv)
    # i = 21389
    # for row in tqdm(csv[21389:]):
    #     i+=1
    #     print(i)
    #     try:
    #         name = 'dataset/2A_images/'+row[0]
    #         img = cv2.imread(name, 0)
    #         img, validate = auto_body_crop(img)
    #         if validate:
    #             cv2.imwrite(f"train/{row[0]}", img)
    #     except:
    #         print("ERRORE :", name)
    #         continue
