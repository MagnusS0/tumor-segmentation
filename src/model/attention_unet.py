import torch
import numpy as np
from monai.networks.nets import AttentionUnet
from monai.transforms import Compose, EnsureChannelFirstd, ScaleIntensityd, LoadImaged, ToTensord

class AttentionUnetModel:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        model = AttentionUnet(
            spatial_dims=2,
            in_channels=4,
            out_channels=4,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2)
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def preprocess(self, img):
        pre_transforms = Compose([
            EnsureChannelFirstd(keys='image'),
            ScaleIntensityd(keys='image'),
            ToTensord(keys='image')
        ])
        return pre_transforms({'image': img})['image']

    def postprocess(self, output):
        # Convert to binary segmentation (white and black pixels)
        output = (output > 0.5).astype(np.uint8) * 255
        return output

    def infer(self, img):
        img = self.preprocess(img)
        img = torch.from_numpy(img).to(self.device).unsqueeze(0)
        with torch.no_grad():
            output = self.model(img)
            output = output.squeeze().cpu().numpy()
        segmentation = self.postprocess(output)
        return segmentation