import torch
from monai.transforms import Compose, LoadImaged, AddChanneld, ScaleIntensityd, ToTensord
from monai.data import DataLoader, Dataset

def load_data(image_files, labels):
    """
    Function to load MIP-PET images and their corresponding labels.
    """
    data_dicts = [{'image': image_name, 'label': label} for image_name, label in zip(image_files, labels)]
    return data_dicts

def create_data_loader(data_dicts, transforms):
    """
    Function to create a data loader using MONAI's Dataset and DataLoader classes.
    """
    dataset = Dataset(data=data_dicts, transform=transforms)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)
    return loader

def evaluate_model(model, data_loader, device):
    """
    Function to evaluate the performance of a model. It calculates the Sørensen-Dice coefficient for each prediction.
    """
    model.eval()
    total_score = 0
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data['image'].to(device), data['label'].to(device)
            outputs = model(inputs)
            total_score += dice_coefficient(outputs, labels).item()
    return total_score / len(data_loader)

def dice_coefficient(pred, target):
    """
    Function to calculate the Sørensen-Dice coefficient.
    """
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)
    m2 = target.view(num, -1)
    intersection = (m1 * m2).sum()
    score = (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)
    return score.mean()

def get_transforms():
    """
    Function to get the MONAI transforms for preprocessing the data.
    """
    transforms = Compose([
        LoadImaged(keys=['image', 'label']),
        AddChanneld(keys=['image', 'label']),
        ScaleIntensityd(keys=['image']),
        ToTensord(keys=['image', 'label'])
    ])
    return transforms