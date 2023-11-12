import unittest
import torch
from models.unet import UNet
from models.vnet import VNet
from models.lfbnet import LFBNet

class TestModels(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.unet = UNet().to(self.device)
        self.vnet = VNet().to(self.device)
        self.lfbnet = LFBNet().to(self.device)

    def test_unet(self):
        # Create a random tensor to represent a batch of images
        x = torch.randn(1, 1, 128, 128, 128).to(self.device)
        # Forward pass
        y = self.unet(x)
        # Check output shape
        self.assertEqual(y.shape, (1, 2, 128, 128, 128))

    def test_vnet(self):
        # Create a random tensor to represent a batch of images
        x = torch.randn(1, 1, 128, 128, 128).to(self.device)
        # Forward pass
        y = self.vnet(x)
        # Check output shape
        self.assertEqual(y.shape, (1, 2, 128, 128, 128))

    def test_lfbnet(self):
        # Create a random tensor to represent a batch of images
        x = torch.randn(1, 1, 128, 128, 128).to(self.device)
        # Forward pass
        y = self.lfbnet(x)
        # Check output shape
        self.assertEqual(y.shape, (1, 2, 128, 128, 128))

if __name__ == '__main__':
    unittest.main()