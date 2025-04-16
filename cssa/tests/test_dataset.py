import unittest
from data.dataset import LLVIPDataset
import torch

class TestLLVIPDataset(unittest.TestCase):

    def setUp(self):
        self.dataset_train = LLVIPDataset(root='data/raw', partition='train')
        self.dataset_val = LLVIPDataset(root='data/raw', partition='val')

    def test_dataset_length(self):
        self.assertTrue(len(self.dataset_train) > 0, "Training dataset should not be empty.")
        self.assertTrue(len(self.dataset_val) > 0, "Validation dataset should not be empty.")

    def test_dataset_getitem(self):
        sample_img, sample_label = self.dataset_train[0]
        self.assertIsInstance(sample_img, torch.Tensor, "Image should be a torch.Tensor.")
        self.assertIsInstance(sample_label, torch.Tensor, "Label should be a torch.Tensor.")
        self.assertEqual(sample_img.dim(), 3, "Image tensor should be 3-dimensional (C,H,W).")

    def test_image_shape_consistency(self):
        for i in range(min(10, len(self.dataset_train))):
            img, _ = self.dataset_train[i]
            self.assertEqual(img.shape[0], 3, "Image should have 3 channels (RGB).")

if __name__ == '__main__':
    unittest.main()
