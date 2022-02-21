
def get_segmented_lungs(raw_im, plot=False):
    from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, \
        reconstruction, binary_closing
    from skimage.measure import label, regionprops, perimeter
    # from skimage.morphology import binary_dilation, binary_opening
    from skimage.filters import roberts, sobel
    # from skimage import measure, feature
    from skimage.segmentation import clear_border, mark_boundaries
    from scipy import ndimage as ndi
    '''
    Original function changes input image (ick!)
    '''
    im = raw_im.copy()
    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    if plot == True:
        f, plots = plt.subplots(9, 1, figsize=(5, 40))

    '''
    Step 1: Convert into a binary image. 
    '''
    binary = im < -400
    if plot == True:
        plots[0].axis('off')
        plots[0].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    if plot == True:
        plots[1].axis('off')
        plots[1].imshow(cleared, cmap=plt.cm.bone)
    '''
    Step 3: Label the image.
    '''
    label_image = label(cleared)
    if plot == True:
        plots[2].axis('off')
        plots[2].imshow(label_image, cmap=plt.cm.bone)
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plots[3].axis('off')
        plots[3].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is 
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plots[4].axis('off')
        plots[4].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is 
    to keep nodules attached to the lung wall.
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)
    if plot == True:
        plots[5].axis('off')
        plots[5].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    if plot == True:
        plots[6].axis('off')
        plots[6].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 8: Superimpose the binary mask on the input image.
    '''
    get_high_vals = binary == 0
    im[get_high_vals] = 0
    if plot == True:
        plots[7].axis('off')
        plots[7].imshow(im, cmap=plt.cm.bone)
        plots[8].axis('off')
        plots[8].imshow(raw_im, cmap=plt.cm.bone)

    plt.show()
    return binary

def segment_cv2(path):
    from skimage import io
    from sklearn import cluster
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np
    import random as rng

    # read input and convert to range 0-1
    image = io.imread(path, as_gray=True) / 255.0
    h, w = image.shape

    # reshape to 1D array
    image_2d = image.reshape(h * w, 1)

    # set number of colors
    numcolors = 3

    # do kmeans processing
    kmeans_cluster = cluster.KMeans(n_clusters=int(numcolors))
    kmeans_cluster.fit(image_2d)
    cluster_centers = kmeans_cluster.cluster_centers_
    cluster_labels = kmeans_cluster.labels_
    # print(cluster_labels, cluster_centers)
    # cluster_centers[0], cluster_centers[1], cluster_centers[2] = 0, 255, 0
    # print(cluster_labels, cluster_centers)
    # need to scale result back to range 0-255
    newimage = cluster_centers[cluster_labels].reshape(h, w)
    newimage = newimage.astype('uint8')
    # display result
    plt.imshow(newimage)
    plt.show()
    plt.hist(newimage.flatten(), bins=80, color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.show()

    # threshold = 250
    # # Detect edges using Canny
    # canny_output = cv2.Canny(newimage, threshold, threshold * 2)
    # # Find contours
    # contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # # Draw contours
    # drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    # for i in range(len(contours)):
    #     color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
    #     cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
    # # Show in a window
    # plt.imshow(drawing)
    # plt.show()
    #
    #
    #
