import argparse
import os

import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from model import UNet 
from data_prepration import SegmentationDataset

from losses import CombinedLoss

def train_model(args):
    image_paths = [os.path.join(args.data_dir, 'images', f) for f in os.listdir(os.path.join(args.data_dir, 'images'))]
    mask_paths = [os.path.join(args.data_dir, 'labels', f) for f in os.listdir(os.path.join(args.data_dir, 'labels'))]

    train_images, val_images, train_masks, val_masks = train_test_split(image_paths, mask_paths, test_size=0.2, random_state=42)

    transform = transforms.Compose([
    transforms.Resize((2048, 2048 )),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(0.456, 0.227)
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

    train_dataset = SegmentationDataset(train_images, train_masks, transform=transform)
    val_dataset = SegmentationDataset(val_images, val_masks, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # print(train_loader[0])
    model = UNet()
    if args.model_weights:
        print('Found resuming weights')
        model.load_state_dict(torch.load(args.model_weights))
        print('training resumed')
    else:
        print("Loading a UNET wih random weights")
    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, )
    
    # multiple gpus implementation
    model = nn.DataParallel(model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        print(f'Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}')
        
        torch.save(model.state_dict(), os.path.join(args.model_save_dir, f'latest_model.pth'))
        
        if (epoch + 1) % args.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(args.model_save_dir, f'model_epoch_ep{epoch + 1}_loss{val_loss / len(val_loader)}.pth'))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--model_save_dir', required=True)
    parser.add_argument('--model_path', required=False, default=None)
    parser.add_argument('--epochs', required=False, default=300)
    parser.add_argument('--batch_size', required=False, default=16)
    parser.add_argument('--save_interval', required=False, default=10)
    
    args = parser.parse_args()

    os.makedirs(args.model_save_dir, exist_ok=True)
    train_model(args)
