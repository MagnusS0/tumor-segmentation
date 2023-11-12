# Import necessary libraries
import monai
from monai.transforms import Compose, LoadImaged, AddChanneld, ScaleIntensityd, ToTensord, RandRotate90d, RandFlipd, RandZoomd, ResizeWithPadOrCropd

# Define keys
keys = ['image', 'label']

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

    # Define the transformations
    transforms = Compose([
        # Add an additional channel
        AddChanneld(keys),
        # Normalize the intensity
        ScaleIntensityd(keys),
        # Resize or crop the image
        ResizeWithPadOrCropd(keys, (96, 96, 96)),
        # Randomly rotate the image
        RandRotate90d(keys, prob=0.5, spatial_axes=(0, 2)),
        # Randomly flip the image
        RandFlipd(keys, spatial_axis=0),
        # Randomly zoom the image
        RandZoomd(keys, prob=0.5, min_zoom=0.9, max_zoom=1.1),
        # Convert the image to a tensor
        ToTensord(keys)
    ])

    # Apply the transformations
    preprocessed_data = transforms(data)

    return preprocessed_data