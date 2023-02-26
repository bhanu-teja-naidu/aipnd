"""
 python predict.py "flowers/test/32/image_05592.jpg" checkpoint.pth --top_k 3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image
import copy
import json
import argparse
import sys
import os
from make_new_model import make_model



def load_model(checkpoint, gpu):
    checkpoint = torch.load(checkpoint)
    model = make_model(checkpoint['arch'], checkpoint['hidden_units'])
    model.classifier = checkpoint["classifier"]
    model.load_state_dict(checkpoint["model_state_dict"])
    model.class_to_idx = checkpoint['class_to_idx']
    if gpu == checkpoint['gpu']:
        device = torch.device('cuda:0' if (torch.cuda.is_available() and gpu) else 'cpu')
        model.to(device)
    else:
        device = torch.device('cuda:0' if (torch.cuda.is_available() and gpu) else 'cpu')
        model.to(device)
    return model
    



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    
    # TODO: Process a PIL image for use in a PyTorch model
    size = 256
    crop_size = 224
    width, height = image.size
    if width > height:
        height = int(height / width * size)
        width = size
    else:
        width = int(width / height * size)
        height = size
    image = image.resize((width, height))
    left = (width - crop_size) / 2
    top = (height - crop_size) / 2
    right = left + crop_size
    bottom = top + crop_size
    image = image.crop((left, top, right, bottom))
    
    np_image = np.array(image) / 255
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - means) / stds
    
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def predict(image_path, model_checkpoint, topk, device):
    ''' 
    Predict the class (or classes) of an image using a trained deep learning model.
    
    Arguments:
        image_path: Path to the image
        model: Trained model
    Returns:
        classes: Top k class numbers.
        probs: Probabilities corresponding to those classes
    '''
    image = Image.open(image_path)

    image = process_image(image)
    image = torch.from_numpy(image)
    image = image.type(torch.FloatTensor)
    image = image.unsqueeze(0)
    image = image.to(device)
    model_checkpoint.to(device)

    with torch.no_grad():
        output = model_checkpoint.forward(image)
        ps = torch.exp(output)
        top_probs, top_indices = ps.topk(topk)
        idx_to_class = {val: key for key, val in model_checkpoint.class_to_idx.items()}
        top_classes = [idx_to_class[index.item()] for index in top_indices[0]]
    top_probs = top_probs.cpu()
    
    return top_probs.numpy()[0], top_classes




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--category_names', type=argparse.FileType('r'))
    parser.add_argument('--top_k', type=int)
    
    args = parser.parse_args()
    image = args.input 
    checkpoint = args.checkpoint
    gpu = args.gpu
    class_file = args.category_names
    top_k = args.top_k
    
    if class_file:
#         with open(class_file, 'r') as f:
        classes = json.load(class_file)
    else:
#         print('Using default file')
        with open('cat_to_name.json', 'r') as f:
            classes = json.load(f)
           
    
    device = torch.device('cuda:0' if (torch.cuda.is_available() and gpu) else 'cpu')

    loaded_model = load_model(checkpoint, gpu)
    probability, prediction = predict(image, loaded_model, top_k or 3, device)
    top_class = [classes[str(x)] for x in  prediction]
#     print(" Actual {classes[image.split('/')[-2]]}")
    print(f"Predicted class: {top_class[0]} with probability: {probability[0]:.4f}")
#     print(device)
    
    if top_k:
        print(f"Top {top_k} most likely classes:")
        for i in range(top_k):
            print(f'{top_class[i]} with prob {probability[i]:.4f}')

    
main()