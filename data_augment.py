import os
import cv2
import imgaug as ia
from imgaug import augmenters as iaa

# Define the directory path where the images to be augmented are stored
input_dir = "crpf"

# Define the augmentation sequence
seq = iaa.Sequential([
    iaa.Fliplr(0.5), # flip horizontally with probability 0.5
    iaa.Affine(rotate=(-10, 10)), # rotate between -10 and 10 degrees
    iaa.GaussianBlur(sigma=(0, 1.0)), # blur with a sigma between 0 and 1.0
    iaa.AdditiveGaussianNoise(scale=(0, 0.1*255)), # add Gaussian noise with a scale between 0 and 0.1*255
])

# Loop through the images and apply the augmentation sequence to each one
for file_name in os.listdir(input_dir):
    input_path = os.path.join(input_dir, file_name)
    image = cv2.imread(input_path)
    # Apply the augmentation sequence multiple times to generate multiple augmented images
    for i in range(5): # generate 5 augmented images for each original image
        image_aug = seq(image=image)
        # Generate a new file name for each augmented image
        output_name = os.path.splitext(file_name)[0] + f"_aug_{i+1}" + os.path.splitext(file_name)[1]
        output_path = os.path.join(input_dir, output_name)
        cv2.imwrite(output_path, image_aug)
