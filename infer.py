import argparse
import os
import cv2 
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from model import UNet
from loaders import testloader
from utils import read_image_files

def test_model(args):

    transform = transforms.Compose([
    transforms.Resize((args.size, args.size)),
    transforms.ToTensor(),
    transforms.Normalize(0.456, 0.227)
    ])

    image_path = read_image_files(args.image_path)
    
    temp_dataset = testloader(image_path, transform)
    shapes = temp_dataset.get_shapes
    temp_loader = DataLoader(temp_dataset, 1)
    
    model = UNet()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for idx, image in enumerate(temp_loader):
            image = image.to(device)
            output = model(image)
            output = (output > 0.5).float()

            output_image = output.cpu().numpy().squeeze()
            output_path = os.path.join(args.result_path, f"output_{idx}.png")
            output_image = cv2.resize(output_image, (shapes[idx][0], shapes[idx][1]), cv2.INTER_AREA)
            cv2.imwrite(output_path, output_image * 255)
            print(f"Saved output {output_path}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', required=True, default=None)
    parser.add_argument('--model_path', required=True, default=None)
    parser.add_argument('--result_path', required=True, default=None)
    parser.add_argument('--size', required=False, default=2048)
    
    args = parser.parse_args()

    os.makedirs(args.result_path, exist_ok=True)
    test_model(args)
