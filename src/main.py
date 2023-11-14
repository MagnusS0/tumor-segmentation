# Import necessary modules
from models.unet import UNet
from models.vnet import VNet
from models.lfbnet import LFBNet
from preprocessing.monai_preprocess import preprocess
from utils.utils import load_data, evaluate_performance

def main():
    # Load MIP-PET images
    images = load_data('data/patients/imgs')

    # Preprocess images using MONAI
    preprocessed_images = preprocess(images)

    # Initialize models
    unet = UNet()
    vnet = VNet()
    lfbnet = LFBNet()

    # List of models
    models = [unet, vnet, lfbnet]

    # Train and evaluate each model
    for model in models:
        # Train model
        model.train(preprocessed_images)

        # Evaluate model performance
        performance = evaluate_performance(model, preprocessed_images)

        # Print model performance
        print(f'{model.__class__.__name__} performance: {performance}')

if __name__ == '__main__':
    main()