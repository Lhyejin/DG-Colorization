# -*- coding: utf-8 -*-
import os
import random
from random import shuffle
import cv2
import numpy as np
from skimage import color
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms

    
class PacsImageDataset(torchvision.datasets.ImageFolder):
    """Custom dataset for loading and pre-processing images."""

    def __init__(self, root, testing=False, domain='photo', H=256, W=256, domain_dict={'photo': 0, 'art_painting': 1, 'cartoon': 2}):
        """Initializes the dataset and loads images.
        Imges should be organized as:
        
            .root/
                class1/
                    img1.jpg
                    img2.jpg
                ..
                classn/
                    imgx.jpg
                    imgy.jpg

        Args:
            root: a directory from which images are loaded
            domain: source domain
            domain_dict: source domain (key-domain name, value-domain label) ditionary
        """
        super().__init__(root=root, loader=Image.open)
        print('root', root)
        self.testing = testing
        self.domain_label = domain_dict[domain]
        if self.testing:
            self.composed = transforms.Compose(
                [transforms.Resize([H, W], interpolation=Image.NEAREST)]
            )
        else:
            self.composed = transforms.Compose([transforms.Resize((H,W), interpolation=Image.NEAREST),
                                                transforms.RandomHorizontalFlip()])

        print('dataset load', domain, len(self.imgs))

    def __getitem__(self, idx):
        """Gets an image in LAB color space.

        Returns:
            Returns a tuple (L, ab, label, name), where:
                L: stands for lightness - it's the net input
                ab: is chrominance - something that the net learns
                label: image class label.
                domain label: domain Label
            Both L and ab are torch.tesnsor
        """
        image, class_label =  super().__getitem__(idx)
        img = self.composed(image.convert('RGB'))
        img = np.array(img)
        lab = color.rgb2lab(img).astype(np.float32)
        lab_t = transforms.ToTensor()(lab)
        x = lab_t[[0], ...] / 50.0 - 1.0 # [-1, 1]
        y = lab_t[[1,2], ...] / 100.0 # [-1, 1]
        return x, y, class_label, self.domain_label

    
    def get_name(self, idx):
        path = os.path.normpath(self.imgs[idx][0])
        name = os.path.basename(path)
        label = os.path.basename(os.path.dirname(path))
        return label + "-" + name

