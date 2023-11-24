import torch
from monai.networks.nets import AttentionUnet
from monai.transforms import Compose, ScaleIntensity, ToTensor, LoadImage, Activations, AsDiscrete, SaveImage
from monai.inferers import SlidingWindowInferer
from monai.data.utils import decollate_batch
from PIL import Image
import numpy as np


class AttentionUnetModel:
    """
    A class representing a model for tumor segmentation using Attention U-Net.

    Attributes:
        device (str): The device to run the model on, defaults to 'cuda' or 'cpu'.
        model (torch.nn.Module): The loaded Attention U-Net model.

    Methods:
        load_model: Loads the model from a given path.
        preprocess_input: Processes input data for the model.
        postprocess_output: Processes the output from the model.
        predict: Runs inference on an input image.

    """

    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initializes an instance of AttentionUnetModel.

        Args:
            model_path (str): The path to the pre-trained model.
            device (str, optional): The device to run the model on. Defaults to 'cuda' if available, else 'cpu'.
        """
        self.device = device
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        """
        Loads the Attention U-Net model from the given path.

        Args:
            model_path (str): The path to the pre-trained model.

        Returns:
            torch.nn.Module: The loaded Attention U-Net model.
        """
        model = AttentionUnet(
            spatial_dims=2,
            in_channels=4,
            out_channels=4,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2)
        )
        # Load pretrained model
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def _preprocess(self, img):
        """
        Preprocesses the input image.

        Args:
            img (str): Path to the input image.

        Returns:
            torch.Tensor: The preprocessed image tensor.
        """
        pre_transforms = Compose([
            ToTensor(),
            LoadImage(image_only=True, ensure_channel_first=True),
            ScaleIntensity()
        ])

        # Add batch dimension and send to device
        return pre_transforms(img).unsqueeze(0).to(self.device)

    def _postprocess(self, output):
        """
        Postprocesses the output from the model.

        Args:
            output (torch.Tensor): The output tensor from the model.

        Returns:
            torch.Tensor: The postprocessed output tensor.
        """
        post_transform = Compose([
            Activations(sigmoid=True),
            AsDiscrete(threshold=0.4),
        ])

        # Remove batch dimension and send to cpu
        return decollate_batch(post_transform(output[0].cpu()))

    def _tensor_to_image(self, tensor):
        """
        Converts a tensor to a PIL Image.

        Args:
            tensor (torch.Tensor): The tensor to convert.

        Returns:
            PIL.Image: The converted PIL Image.
        """
        array = tensor[0].numpy()
        array = np.transpose(array, (1, 0))  # Swap the axes
        return Image.fromarray((array * 255).astype(np.uint8))

    def infer(self, img, roi_size=(96, 96), sw_batch_size=4):
        """
        Run inference on an input image.

        Args:
            img (str): Path to the input image.
            roi_size (tuple, optional): Size of the ROI to use for inference. Defaults to (96, 96).
            sw_batch_size (int, optional): Batch size for sliding window inference. Defaults to 4.

        Returns:
            PIL.Image: The segmented image.
        """
        img = self._preprocess(img)
        inferer = SlidingWindowInferer(roi_size, sw_batch_size)

        with torch.no_grad():
            output = inferer(inputs=img, network=self.model)

        segmentation = self._postprocess(output)

        return self._tensor_to_image(segmentation)


if __name__ == "__main__":
    model = AttentionUnetModel(
        model_path="./src/model/best_metric_model_segmentation2d_dict.pth")
    seg = model.infer(img="./data/patients/imgs/patient_150.png")
    print(seg)
