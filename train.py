"""
usage: python train.py "flowers" --learning_rate 0.001 --hidden_units 4096 --epochs 3 --gpu
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



batch_size=16

def make_datasetand_loaders(data_dir):

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms = {
        train_dir: transforms.Compose([
            transforms.RandomRotation(degrees=30),
            transforms.RandomResizedCrop(size=224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        valid_dir: transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        test_dir: transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }


    # TODO: Load the datasets with ImageFolder
    image_datasets = {dirr: datasets.ImageFolder(root=dirr, transform=data_transforms[dirr]) for dirr in [train_dir, valid_dir, test_dir]}
    
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    
    dataloaders = {
        train_dir: DataLoader(image_datasets[train_dir], batch_size=batch_size, shuffle=True),
        valid_dir: DataLoader(image_datasets[valid_dir], batch_size=batch_size, shuffle=False),
        test_dir: DataLoader(image_datasets[test_dir], batch_size=batch_size, shuffle=False)
    }
    return image_datasets, dataloaders









def make_model(arch='vgg16', hidden_units=4096):
    try:
        model = getattr(torchvision.models, arch)(pretrained=True)
    except ModelNotFoundError:
        print('Trying with vgg16')
        model = getattr(torchvision.models, 'vgg16')(pretrained=True)
    
    for parameter in model.parameters():
        parameter.requires_grad=False
        
    if(hidden_units > 5):
        hidden_units = hidden_units
    else:
        print("Given hidden units are not valid. Trying with 4096 hidden units")
        hidden_units = 4096

    classifier = nn.Sequential(
        nn.Linear(25088, hidden_units),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )

    model.classifier = classifier
    return model

    


def valid_model(model, criterion, valid_loader, device):
    running_loss, running_accuracy = 0.0, 0.0
    size= 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            size += labels.size(0)
            running_accuracy += (preds == labels).sum().item()
    val_loss = running_loss / len(valid_loader)
#     print(len(valid_loader))
    val_acc = 100.0 * running_accuracy / (len(valid_loader) * batch_size)
    return val_loss, val_acc



def train_model(model, criterion, optimizer, train_loader, valid_loader, num_epochs, device):
    best_acc = 0.0
    valid_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    step = 0
    print_every=50
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)



        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            step += 1
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data)
            
            if step % print_every == 0:
                valid_loss, valid_acc = valid_model(model, criterion, valid_loader, device)
                print("Training loss: {:.3f}".format(running_loss/print_every),
                  "Training Accuracy: {:.3f}".format(100 * running_corrects.double()/(print_every * batch_size)),
                  "Valid loss : {:.3f}".format(valid_loss),
                  "Valid Accuracy : {:.3f}".format(valid_acc))
                running_loss = 0
                running_corrects = 0
                model.train()
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model_wts = copy.deepcopy(model.state_dict())
    print("Best acc: {:.4f}".format(best_acc))
    
    model.load_state_dict(best_model_wts)
    return model




def test_model(model, test_loader, criterion, device):
    running_loss, running_accuracy = 0.0, 0.0
    size= 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels.detach())
            running_loss += loss.item()
            size += labels.size(0)
            running_accuracy += (preds == labels).sum().item()
    test_loss = running_loss / len(test_loader)
    test_acc = 100.0 * running_accuracy / size
    return test_loss, test_acc



def main():
    
    
       
    
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, default='flowers')
    parser.add_argument('--save_dir', default='./')
    parser.add_argument('--arch', default='vgg16')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--hidden_units', type=int, default=4096)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--gpu', default=False, action='store_true')
    
    args = parser.parse_args()
    data_dir = args.data_dir
    num_epochs = args.epochs
    h_units = args.hidden_units
    lr = args.learning_rate
    arch = args.arch
    out_dir = args.save_dir
    gpu = args.gpu
    
    if os.path.exists(data_dir):
        data_dir = data_dir
    else:
        print("data_dir not valid. Attempting to use default flowers dir")
        data_dir = 'flowers'
        
    batch_size=16
    image_datasets, dataloaders = make_datasetand_loaders(data_dir)
    
    model = make_model(arch, h_units)
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    
    device = torch.device('cuda:0' if (torch.cuda.is_available() and gpu) else 'cpu')
    model = model.to(device)

    criterion = nn.NLLLoss()
#     print(len(dataloaders[data_dir+'/train']), len(dataloaders[data_dir+'/valid']))

    # best hyper parameters: lr=0.001, hidden_units=4096, epochs=3
    
    model_trained = train_model(model, criterion, optimizer, dataloaders[data_dir+'/train'], dataloaders[data_dir+'/valid'], num_epochs, device)

    test_loss, test_acc = test_model(model_trained, dataloaders[data_dir+'/test'], criterion, device)
    print(f"Test loss: {test_loss}, Test Acc: {test_acc}")


    model_trained.class_to_idx = image_datasets[data_dir+'/train'].class_to_idx
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    checkpoint_path = os.path.join(out_dir, 'checkpoint.pth')


    torch.save({
        'epoch': num_epochs,
        'arch':arch,
        'hidden_units': h_units,
        'lr': lr,
        'saved_path': out_dir,
        'model_state_dict': model_trained.state_dict(),
        'classifier': model_trained.classifier,
        'optimizer_state_dict' : optimizer.state_dict(),
        'class_to_idx': model_trained.class_to_idx,
        'gpu': gpu,
    }, checkpoint_path)
    
    
main()