import nibabel as nib

def transform_to_hu(medical_image, image):

    intercept = medical_image.RescaleIntercept

    slope = medical_image.RescaleSlope

    hu_image = image * slope + intercept



    return hu_image

def window_image(image, window_center, window_width):

    img_min = window_center - window_width // 2

    img_max = window_center + window_width // 2

    window_image = image.copy()

    window_image[window_image < img_min] = img_min

    window_image[window_image > img_max] = img_max



    return window_image

medical_image = nib.load('dataset/unife/POS/29_000.nii.gz')
image = medical_image.get_fdata()
print(medical_image.header)
hu_image = transform_to_hu(medical_image,image[150])

brain_image = window_image(hu_image, 40, 80)

bone_image = window_image(hu_image, 400, 1000)