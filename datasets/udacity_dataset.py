import pandas as pd
import numpy as np
import torch
import torchvision
from torchvision import transforms as tf
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
import random
import os

class UdacityDataset(data.Dataset):
    def __init__(self, root, dataframe, transforms = None):
        super().__init__()
        self.root = root
        self.df = dataframe
        self.transforms = transforms
        self.load_data()
        
    def load_data(self):
        self.fns = []
        for idx, row in self.df.iterrows():
            center = os.path.join(self.root, os.path.basename(row[0]))
            left = os.path.join(self.root, os.path.basename(row[1]))
            right = os.path.join(self.root, os.path.basename(row[2]))
            steering = row[3]
            throttle = row[4]
           
            self.fns.append((center,steering,throttle))
            self.fns.append((left,steering+0.2,throttle))
            self.fns.append((right,steering-0.2,throttle))
            
    def __getitem__(self, idx):
        img, steering, throttle = self.fns[idx]
        img = Image.open(img)

        if self.transforms is not None:
            
            if random.randint(1,10)/10 <= 0.5:
                img = np.flip(np.array(img), axis=1)
                steering *= -1
                img = Image.fromarray(img)
            img = self.transforms(img)   
            
        steering = torch.FloatTensor([steering])
        throttle = torch.FloatTensor([throttle])
        
        return img, steering
    
    def __len__(self):
        return len(self.fns)
    
    def collate_fn(self, batch):
        imgs = torch.stack([i[0] for i in batch])
        labels = torch.stack([i[1] for i in batch])
        return {
            'imgs': imgs,
            'labels': labels
        }

    def __str__(self):
        s = "Udacity Dataset\n"
        line = "-------------------------------\n"
        s1 = "Number of samples: " + str(len(self.fns)) + '\n'
        return s + line + s1 