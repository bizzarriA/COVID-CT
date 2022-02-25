from colorsys import yiq_to_rgb
import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
import pandas as pd
from tqdm import tqdm


def read_csv(base_path):
    train_df = pd.read_csv(base_path + 'train_COVIDx_CT-2A.txt', sep=" ", header=None)
    train_df.columns = ['filename', 'label', 'xmin', 'ymin', 'xmax', 'ymax']
    # train_df = train_df.drop(['xmin', 'ymin', 'xmax', 'ymax'], axis=1)
    val_df = pd.read_csv(base_path + 'val_COVIDx_CT-2A.txt', sep=" ", header=None)
    val_df.columns = ['filename', 'label', 'xmin', 'ymin', 'xmax', 'ymax']
    # val_df = val_df.drop(['xmin', 'ymin', 'xmax', 'ymax'], axis=1)

    test_df = pd.read_csv(base_path + 'test_COVIDx_CT-2A.txt', sep=" ", header=None)
    test_df.columns = ['filename', 'label', 'xmin', 'ymin', 'xmax', 'ymax']
    # test_df = test_df.drop(['xmin', 'ymin', 'xmax', 'ymax'], axis=1)

    image_path = base_path + '2A_images/'  # directory path
    train_df['filename'] = image_path + train_df['filename']
    val_df['filename'] = image_path + val_df['filename']
    test_df['filename'] = image_path + test_df['filename']
    print("train: ",len(train_df),"val: ", len(val_df),"test: ", len(test_df))
    return train_df, test_df, val_df


def plot_img(data):
    # Select cases to view
    np.random.seed(27)
    indices = np.random.choice(list(range(len(data))), 9)

    # Show a grid of 9 images
    fig, axes = plt.subplots(3, 3, figsize=(16, 16))
    classes = data["label"]
    class_names = ('Normal', 'Pneumonia', 'COVID-19')
    for index, ax in zip(indices, axes.ravel()):
        # Load the CT image
        image = cv2.imread(data["filename"][index], cv2.IMREAD_UNCHANGED)
        # Overlay the bounding box
        image = np.stack([image] * 3, axis=-1)  # make image 3-channel
        bbox = (data["xmin"][index], data["ymin"][index], data["xmax"][index], data["ymax"][index])
        cv2.rectangle(image, bbox[:2], bbox[2:], color=(255, 0, 0), thickness=3)
        print(image.min(), image.max())
        # Display
        cls = classes[index]
        #     plt.figure()
        ax.imshow(image)
        ax.set_title('Class: {} ({})'.format(class_names[cls], cls))
    plt.show()

def read_slice(base_path):
    classe = 'png/'
    patientes_path = os.listdir(base_path+classe)
    csv = pd.read_csv(base_path+'test_set_unife.csv', sep=';')
    names = []
    y_true = []
    patientes = []
    immagini_png = []
    label_tot = []
    for path in tqdm(patientes_path):
        name = path + "_0.nii.gz"
        names.append(path)
        label = csv[csv["filename"]==name]["label"].values.tolist()
        y_true.append(label)
        scans_path = os.listdir(base_path + classe + path)
        patient = []
        centro = len(scans_path) // 2
        for scan_path in scans_path[centro-10:centro+10]:
            try:
                scan = cv2.imread(base_path + classe + path + '/' + scan_path, 0) / 255.
                scan = cv2.resize(scan, (256, 256))
                scan = np.expand_dims(scan, axis=0)
                scan = np.expand_dims(scan, axis=-1)
            except:
                continue
            immagini_png.append(scan)
            label_tot.append(label)
            patient.append(scan)
        patientes.append(patient)
        
    print("[INFO] Numero pazienti: {} - Numero totale immagini: {} - Numero totale etichette: {}".format(len(patientes), len(immagini_png), len(label_tot)))
    return patientes, y_true, immagini_png, label_tot
    
    
    

def convert_nifti():
    base_path = 'dataset/unife/'
    lista = os.listdir(base_path+'POS')
    for p in lista:
        print("[INFO]", p)
        directory = base_path + 'png/' + p[:-7]
        print("[INFO]", directory)
        if not os.path.exists(directory):
            os.makedirs(directory)
        os.system("python utils/nii2png.py -i {0}POS/{1} -o {2}".format(base_path, p, directory))
        

if __name__ == '__main__':
    convert_nifti()
