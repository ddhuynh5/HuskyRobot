""" Custom PyTorch Dataset Class """

import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image


class HuskyDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        image_path = self.dataframe.iloc[index]['image']
        measurements = self.dataframe.iloc[index]['measurements']
        
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)
        
        linear = torch.tensor([measurements['linear']['x'], measurements['linear']['y'], measurements['linear']['z']])
        angular = torch.tensor([measurements['angular']['x'], measurements['angular']['y'], measurements['angular']['z']])
        measurements_tensor = torch.cat([linear, angular])
        
        return image, measurements_tensor