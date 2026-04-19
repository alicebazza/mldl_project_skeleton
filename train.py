from __future__ import print_function, division

from typing import Mapping, Union, Optional

import os
import random
import argparse
import numpy as np
from tqdm.notebook import tqdm
import plotly.graph_objects as go

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms

torch.manual_seed(42)
np.random.seed(42)
random.seed(0)

torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from models.models import FC2Layer, CNN
from utils.data import get_dataloaders
from utils.utils import count_parameters
from utils.training import fit, get_model_optimizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# PARAMETRI
train_loader, test_loader = get_dataloaders()
x, yy = next(iter(train_loader))

n_channels = x.shape[1]
input_size_w = x.shape[2]
input_size_h = x.shape[3]
input_size = input_size_w * input_size_h

output_size = yy.max().item() + 1
output_classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

epochs = 4
n_hidden = 9
n_features = 6
models_accuracy = {}

# FULLY CONNECTED LAYER
model_fnn = FC2Layer(input_size, n_channels, n_hidden, output_size)
model_fnn.to(device)
optimizer = get_model_optimizer(model_fnn)

print(f'Number of parameters: {count_parameters(model_fnn)}')

fit(epochs=epochs,
    train_dl=train_loader,
    test_dl=test_loader,
    model=model_fnn,
    opt=optimizer,
    tag='fnn',
    device=device)

# CONVOLUTIONAL LAYER
model_cnn = CNN(input_size, n_channels, n_features, output_size)
model_cnn.to(device)
optimizer = get_model_optimizer(model_cnn)

print(f'Number of parameters: {count_parameters(model_cnn)}')

fit(epochs=epochs,
    train_dl=train_loader,
    test_dl=test_loader,
    model=model_cnn,
    opt=optimizer,
    tag='cnn',
    device=device)
