import os

import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image


class testloader(Dataset):
    
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform
    
    @property
    def get_shapes(self):
        shapes = []
        if os.path.isdir(self.image_path):
            for file in os.listdir(self.image_path):
                image = cv2.imread(os.path.join(self.image_path, file))
                shapes.append((image.shape[1], image.shape[0]))
        else:
            img = cv2.imread(self.image_path)
            shapes = [(img.shape[1], img.shape[0])]
        return shapes
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        image = Image.open(self.image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

class trainloader(Dataset):
    
    def __init__(self, image_dir, mask_dir, transform=None):
        self.transform = transform
        self.image_names = image_dir
        self.mask_names = mask_dir

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_path = self.image_names[idx]
        mask_path = self.mask_names[idx]
        
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Convert mask to grayscale
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
                
        # Ensure mask is a binary tensor
        mask = torch.where(mask > 0, 1, 0).type(torch.FloatTensor)
        
        return image, mask
