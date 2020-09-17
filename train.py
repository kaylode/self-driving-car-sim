from utils.getter import *
from datasets.udacity_dataset import UdacityDataset
from models.regressor import Regressor
from losses.mseloss import MSELoss
import torchvision.transforms as tf
import os 
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.utils.data as data
import torch
import torch.nn as nn

train_transforms = tf.Compose([
    tf.RandomAffine(0, translate=(0,0.2)),
    tf.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0),
    tf.ToTensor(),
    tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = tf.Compose([
    tf.ToTensor(),
    tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if __name__ == "__main__":
    device = torch.device("cuda")
    print("Using", device)

    root = 'datasets/udacity'
    csv_file = os.path.join(root, 'driving_log.csv')
    img_folder = os.path.join(root, 'IMG')

    table = pd.read_csv(csv_file)
    traindata, valdata = train_test_split(table, train_size=0.8, random_state=2020)
    
    # Dataloader
    trainset = UdacityDataset(img_folder, traindata, transforms=train_transforms)
    valset = UdacityDataset(img_folder, valdata, transforms=val_transforms)
    trainloader = data.DataLoader(trainset, batch_size=16,collate_fn=trainset.collate_fn)
    valloader = data.DataLoader(valset, batch_size=16, collate_fn=trainset.collate_fn)
    print(trainset)
    print(valset)
    

    NUM_CLASSES = 1
    criterion = MSELoss()
    optimizer = torch.optim.Adam
    
    model = Regressor(
                    n_classes = NUM_CLASSES,
                    optim_params = {'lr': 1e-3},
                    criterion= criterion, 
                    optimizer= optimizer,
                    device = device)
    
    trainer = Trainer(model,
                     trainloader, 
                     valloader,
                     checkpoint = Checkpoint(save_per_epoch=5, path = 'weights/udacity'),
                     logger = Logger(log_dir='loggers/runs/udacity'),
                     scheduler = StepLR(model.optimizer, step_size=25, gamma=0.1),
                     evaluate_per_epoch = 2)
    
    print(trainer)
    
    trainer.fit(num_epochs=30, print_per_iter=10)

