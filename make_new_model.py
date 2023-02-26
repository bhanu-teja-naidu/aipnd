import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision


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

    