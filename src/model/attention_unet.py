import torch
from monai.networks.nets import AttentionUnet
from monai.transforms import Compose, ScaleIntensity, ToTensor, LoadImage, Activations, AsDiscrete, SaveImage
from monai.inferers import SlidingWindowInferer
from monai.data.utils import decollate_batch


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
        self.device = device
        self.model = self._load_model(model_path)

    # Set the model with same parameters as in training
    def _load_model(self, model_path):
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

    # Turn input image into tensor
    def _preprocess(self, img):
        pre_transforms = Compose([
            ToTensor(),
            LoadImage(image_only=True, ensure_channel_first=True),
            ScaleIntensity()
        ])
        
        return pre_transforms(img).unsqueeze(0).to(self.device) # Add batch dimension and send to device

    def _postprocess(self, output):
        post_transform = Compose([
            Activations(sigmoid=True),
            AsDiscrete(threshold=0.4),
        ])

        return decollate_batch(post_transform(output[0].cpu())) # Remove batch dimension and send to cpu
    
    def infer(self, img, roi_size=(96, 96), sw_batch_size=4, output_dir="./output"):
        """
        Run inference on an input image.

        Args:
            img (str): Path to the input image.
            roi_size (tuple, optional): Size of the ROI to use for inference. Defaults to (96, 96).
            sw_batch_size (int, optional): Batch size for sliding window inference. Defaults to 4.
            output_dir (str, optional): Directory to save the output images. Defaults to "./output".

        Returns:
            List[str]: Paths of the saved segmentation images.
        """
        img = self._preprocess(img)
        inferer = SlidingWindowInferer(roi_size, sw_batch_size)

        with torch.no_grad():
            output = inferer(inputs=img, network=self.model)

        segmentation = self._postprocess(output)

        saver = SaveImage(
            output_dir=output_dir,
            output_ext=".png",
            output_postfix="seg",
            scale=255
        )
        return saver(segmentation)


if __name__ == "__main__":
    model = AttentionUnetModel(model_path="./src/model/best_metric_model_segmentation2d_dict (4).pth")
    model.infer(img="./data/patients/imgs/patient_150.png")