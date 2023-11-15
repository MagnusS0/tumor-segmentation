import unittest
import numpy as np
import torch
from src.model.attention_unet import AttentionUnetModel


class TestAttentionUnetModel(unittest.TestCase):
    def setUp(self):
        self.model_path = 'path/to/model.pth'
        self.device = 'cpu'
        self.model = AttentionUnetModel(self.model_path, self.device)

    def test_preprocess(self):
        # Test preprocessing function
        img = np.random.rand(4, 256, 256)
        preprocessed_img = self.model.preprocess(img)
        self.assertEqual(preprocessed_img.shape, (4, 256, 256))
        self.assertTrue(torch.is_tensor(preprocessed_img))

    def test_postprocess(self):
        # Test postprocessing function
        output = np.random.rand(4, 256, 256)
        postprocessed_output = self.model.postprocess(output)
        self.assertEqual(postprocessed_output.shape, (4, 256, 256))
        self.assertTrue(np.all(np.logical_or(postprocessed_output == 0, postprocessed_output == 255)))

    def test_infer(self):
        # Test inference function
        img = np.random.rand(4, 256, 256)
        segmentation = self.model.infer(img)
        self.assertEqual(segmentation.shape, (4, 256, 256))
        self.assertTrue(np.all(np.logical_or(segmentation == 0, segmentation == 255)))

if __name__ == '__main__':
    unittest.main()