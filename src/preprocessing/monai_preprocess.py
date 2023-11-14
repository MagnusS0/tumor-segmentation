# Import necessary libraries
import monai
import numpy as np
from monai.transforms import Compose, LoadImaged, AddChanneld, ScaleIntensityd, ToTensord, ResizeWithPadOrCropd

# Define keys
keys = ['imgs', 'label']


# Define the preprocessing function
def preprocess(data_path, keys):
    """
    This function takes in a path to MIP-PET images and applies normalization, resizing, and augmentation.
    Args:
        data_path (str): Path to the MIP-PET images.
        keys (list): List of keys to use for the dictionary of image data.
    Returns:
        preprocessed_data (list): List of preprocessed image data.
    """
    # Load the data
    data = LoadImaged(keys)(data_path)

    # Convert 3D PET images to 2D MIP-PET images
    data['imgs'] = np.max(data['imgs'], axis=1)

    # Define the transformations
    transforms = Compose([
        # Add an additional channel
        AddChanneld(keys),
        # Normalize the intensity
        ScaleIntensityd(keys),
        # Resize or crop the image
        ResizeWithPadOrCropd(keys, (400, 991)),
        # Convert the image to a tensor
        ToTensord(keys)
    ])

    # Apply the transformations
    preprocessed_data = transforms(data)

    return preprocessed_data