
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf
from tqdm import tqdm



def read_csv():
    base_path = 'dataset/'
    train_df = pd.read_csv(base_path + 'train_COVIDx_CT-2A.txt', sep=" ", header=None)
    train_df.columns = ['filename', 'label', 'xmin', 'ymin', 'xmax', 'ymax']
    train_df = train_df.drop(['xmin', 'ymin', 'xmax', 'ymax'], axis=1)
    
    val_df = pd.read_csv(base_path + 'val_COVIDx_CT-2A.txt', sep=" ", header=None)
    val_df.columns = ['filename', 'label', 'xmin', 'ymin', 'xmax', 'ymax']
    val_df = val_df.drop(['xmin', 'ymin', 'xmax', 'ymax'], axis=1)

    test_df = pd.read_csv(base_path + 'test_COVIDx_CT-2A.txt', sep=" ", header=None)
    test_df.columns = ['filename', 'label', 'xmin', 'ymin', 'xmax', 'ymax']
    test_df = test_df.drop(['xmin', 'ymin', 'xmax', 'ymax'], axis=1)

    new_train = pd.read_csv('total_test_data.csv')
    new_train = new_train.iloc[:,1:]
    #new_train.columns = ['filename', 'label']
    #new_train = new_train[new_train['filename'].str.contains('test') == False]
    train_df['filename'] = 'dataset/2A_images/' + train_df['filename']
    val_df['filename'] = 'dataset/2A_images/' + val_df['filename']
    test_df['filename'] = 'dataset/2A_images/' + test_df['filename']
    # print(test_df)
    # ## read unife and append
    # unife_df = pd.read_csv(base_path+'unife.csv')
    # unife_df = unife_df[unife_df['label']!=0]
    # train_df = pd.concat([train_df, unife_df[unife_df['split']=='train']])
    # val_df = pd.concat([val_df, unife_df[unife_df['split']=='val']])
    # test_df = pd.concat([test_df, unife_df[unife_df['split']=='val']])
    
    
    print("train: ",len(train_df),"val: ", len(val_df),"test: ", len(test_df))
    return train_df, test_df, val_df, new_train


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

def read_slice(base_path, shuffle=False):
    classe = 'png/'
    patientes_path = os.listdir(base_path+classe)
    if shuffle:
        random.shuffle(patientes_path)
    print(len(patientes_path))
    csv = pd.read_csv(base_path+'test_set_unife.csv')
    names = []
    y_true = []
    patientes = []
    immagini_png = []
    label_tot = []
    for path in tqdm(patientes_path):
        try:
            # names.append(path)
            label = csv.loc[csv["filename"]==path]["label"].item()
            y_true.append(label)
            scans_path = os.listdir(base_path + classe + path)
            patient = []
            centro = len(scans_path) // 2
            if label == 0:
                n = 60
            elif label == 1:
                n = 60
            elif label == 2:
                n = 60
            for scan_path in scans_path[centro-n:centro+n]:
                scan = load_and_process(path=base_path + classe + path + '/' + scan_path)
                immagini_png.append(scan)
                label_tot.append(label)
                patient.append(scan)
                names.append(path+'_'+scan_path)
            patientes.append(patient)
        except:
            print("[ERROR] read path:", path)        
    print("[INFO] Numero pazienti: {} - Numero totale immagini: {} - Numero totale etichette: {}".format(len(patientes), len(immagini_png), len(label_tot)))
    return names, immagini_png, label_tot
    
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
        


def load_and_process(row=None, path=None):
    # Load image
    # print(row)
    if row is not None:
        path = row[0]
    # print("[INFO] ", path, label, bbox)
    SIZE = 256
    image = cv2.imread(path, 0)
    if row is not None:
        bbox = ([int(row[2]), int(row[3]), int(row[4]), int(row[5])])
        image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    image = image / 255.0
    image = cv2.resize(image, (SIZE, SIZE))
    image = tf.expand_dims(image, axis=-1)

    return image



if __name__=="__main__":
    import numpy as np
    import pandas as pd
    import os
    import cv2
    from random import sample
    from preprocessing import auto_body_crop


    test_df = pd.read_csv('dataset/train_COVIDx_CT-2A.txt', sep=' ')
    test_df.columns = ['filename', 'label', 'xmin', 'ymin', 'xmax', 'ymax']
    test_df = test_df.drop(['xmin', 'ymin', 'xmax', 'ymax'], axis=1)
    test_df = test_df[test_df['filename'].str.contains('HUST')]
    test_df = test_df.sample(n=10000)
    id = []
    labels = []
    
    

    
    #test = os.listdir('test/')
    base_path = 'dataset/2A_images/'
    dest_path = 'train/'

    for i, row in test_df.iterrows():
        try: 
            # HUST-Patient1-0062.png --> Patient1_0062.png
            #name = row[0].replace("HUST-", "")
            name_old = row[0].replace("-", "_")
            name = name_old.replace('_', '-', 1)
            # os.system(f"mv {name_old} {name}")
            img = cv2.imread(base_path + row[0], 0)
            img, validate = auto_body_crop(img)
            if validate:
                cv2.imwrite('new_train/' + name, img)
            name = dest_path + name
            print(name)
            labels.append(row[1])
            id.append(name)
        except:
            print("ERRORE file: ", name)
            continue
    id = np.array(id).flatten()
    labels = np.array(labels).flatten()
    print(np.shape(id), np.shape(labels))
    new_data = pd.DataFrame({'filename': id, 'label': labels})
    test_data = pd.read_csv('train_data.csv')
    test_data = test_data.append(new_data, ignore_index=True)
    test_data.to_csv('total_train_data.csv')
    
    print(id[:10])
    
    
    

