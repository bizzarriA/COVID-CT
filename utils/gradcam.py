import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from preprocessing import auto_body_crop

def load_and_preprocess(image_files, width=256, height=256):
    """Loads and preprocesses images for inference"""
    images = []
    for image_file in image_files:
        # Load and crop image
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        # image, validate = auto_body_crop(image)
        # if validate:
        image = cv2.resize(image, (width, height), cv2.INTER_CUBIC)

        # Convert to float in range [0, 1] and stack to 3-channel
        image = image.astype(np.float32) / 255.0
        # image = np.stack((image, image, image), axis=-1)
        
        # Add to image set
        images.append(image)
    
    return np.array(images)

    

def run_gradcam(model, layerName, image, classIdx, eps=1e-8):
    # construct our gradient model by supplying (1) the inputs
    # to our pre-trained model, (2) the output of the (presumably)
    # final 4D layer in the network, and (3) the output of the
    # softmax activations from the model
    # print("[INFO] Layer Name: ", layerName)
    gradModel = tf.keras.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layerName).output,
            model.output])

    # record operations for automatic differentiation
    with tf.GradientTape() as tape:
        # cast the image tensor to a float-32 data type, pass the
        # image through the gradient model, and grab the loss
        # associated with the specific class index
        inputs = tf.cast(image, tf.float32)
        (convOutputs, predictions) = gradModel(inputs)
        loss = predictions[:, classIdx]
    # use automatic differentiation to compute the gradients
    grads = tape.gradient(loss, convOutputs)
    # compute the guided gradients
    castConvOutputs = tf.cast(convOutputs > 0, "float32")
    castGrads = tf.cast(grads > 0, "float32")
    guidedGrads = castConvOutputs * castGrads * grads
    # the convolution and guided gradients have a batch dimension
    # (which we don't need) so let's grab the volume itself and
    # discard the batch
    convOutputs = convOutputs[0]
    guidedGrads = guidedGrads[0]
    # compute the average of the gradient values, and using them
    # as weights, compute the ponderation of the filters with
    # respect to the weights
    weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
    # grab the spatial dimensions of the input image and resize
    # the output class activation map to match the input image
    # dimensions
    (w, h) = (image.shape[2], image.shape[1])
    heatmap = cv2.resize(cam.numpy(), (w, h))

    # Convert to [0, 1] range
    heatmap = np.maximum(heatmap, 0)/np.max(heatmap)

    # Resize to image dimensions
    heatmap = cv2.resize(heatmap, (image.shape[2], image.shape[1]))

    return heatmap


def find_target_layer(model):
    # attempt to find the final convolutional layer in the network
    # by looping over the layers of the network in reverse order
    for layer in reversed(model.layers):
        # check to see if the layer has a 4D output
        if len(layer.output_shape) == 4:
            return layer.name
    # otherwise, we could not find a 4D layer so the GradCAM
    # algorithm cannot be applied
    raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

def stacked_bar(ax, probs):
    """Creates a stacked bar graph of slice-wise predictions"""
    x = list(range(probs.shape[0]))
    width = 0.8
    ax.bar(x, probs[:, 0], width, color='g')
    ax.bar(x, probs[:, 1], width, bottom=probs[:, 0], color='r')
    ax.bar(x, probs[:, 2], width, bottom=probs[:, :2].sum(axis=1), color='b')
    ax.set_ylabel('Confidence')
    ax.set_xlabel('Slice Index')
    ax.set_title('Class Confidences by Slice')
    ax.legend(CLASS_NAMES, loc='upper right')
    
# Model directory, metagraph file name, and checkpoint name
MODEL_DIR = 'models/prova'
LAYERNAME = 'batch_normalization_3'

# Class names, in order of index
CLASS_NAMES = ('0_normal', '1_common', '2_covid')
CLASSE = CLASS_NAMES[2]

# Load Model
model = tf.keras.models.load_model("model/model_jpeg_18_03_all_image") #'model/model_18_03_crop_img')
model.summary()
# Select image file

base_path = 'nuovi/test/Patient 23/CT/' #dataset/2A_images/'
# image_files = os.listdir(base_path)
# image_files = ['LIDC-IDRI-0273-1.3.6.1.4.1.14519.5.2.1.6279.6001.268992195564407418480563388746-0093.png',
#               'CP_5_3509_0130.png',
#               'HUST-Patient1314-0348.png']
csv = pd.read_csv('dataset/test_COVIDx_CT-2A.txt', sep=' ', header=None)
csv.columns = ['filename', 'label', 'xmin', 'ymin', 'xmax', 'ymax']
images = []
hmaps = []
idxs = []
confidences = []
csv = csv.sample(n=100)
image_files = np.array(csv['filename'])
image_files = os.listdir(base_path)
for image_file in image_files:
    # Prepare imags
    try:
        image = load_and_preprocess([base_path + image_file])
        preds = model.predict(image, batch_size=1)
        classIdx = np.argmax(preds)

        # Run Grad-CAM
        if LAYERNAME is None:
                    LAYERNAME = find_target_layer(model)
        heatmap = run_gradcam(
            model, LAYERNAME, image, classIdx)
        
        images.append(image)
        idxs.append(classIdx)
        confidences.append(preds[0][classIdx])
        hmaps.append(heatmap)

        # Show image
        fig = plt.plot(figsize=(10, 5))
        plt.subplots_adjust(hspace=0.01)
        plt.imshow(image[0])
        plt.imshow(heatmap, cmap='jet', alpha=0.4)
        plt.savefig(f"heatmap/normal_{classIdx}_{image_file}")
    except:
        continue
print('**DISCLAIMER**')
print('Do not use this prediction for self-diagnosis. '
    'You should check with your local authorities for '
    'the latest advice on seeking medical assistance.')
