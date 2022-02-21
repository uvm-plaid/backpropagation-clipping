
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import pickle
import itertools

import time
from torchvision import datasets, transforms

import upstream_clipping as uc

from data_loader import DPDataLoader

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

dataset1 = datasets.MNIST('data', train=True, download=True,
                   transform=transform)
dataset2 = datasets.MNIST('data', train=False,
                   transform=transform)

n_features = 784
n_classes = 10

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.network = nn.Sequential(
            torch.nn.Conv2d(1, 16, 8, 2, bias=False),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 1), 
            torch.nn.Conv2d(16, 32, 4, 2, bias=False),
            torch.nn.ReLU(), 
            torch.nn.MaxPool2d(2, 1), 
            torch.nn.Flatten(), 
            torch.nn.Linear(288, 32, bias=False),
            torch.nn.ReLU(), 
            torch.nn.Linear(32, 10, bias=False),
            nn.LogSoftmax(dim=1))

        self.l1_clip = [self.network[1],
                       self.network[4]]
        self.l2_clip = [self.network[8],
                       self.network[10]]
        self.input_clip = [self.network[0],
                          self.network[3],
                          self.network[7],
                          self.network[9]]
        
    def forward(self, x):
        return self.network(x)


# CONFIGURATIONS
# delta is fixed to 10e-5
# decent hyperparameters for all configurations:
batches = [256]
epochses = [5]
grad_clips = [1e-7]
inp_clips = [3]

# EPSILON = 3 configuration
# reaches ~96% accuracy
epsilons = [3]

# EPSILON = 1 configuration
# reaches ~95% accuracy
epsilons = [1]

# EPSILON = 0.2 configuration
# reaches ~87% accuracy
epsilons = [0.2]

# EPSILON = 0.1 configuration
# reaches ~79% accuracy
epsilons = [0.1]

for BATCH_SIZE, epochs, grad_clip, inp_clip, target_eps \
    in itertools.product(batches, epochses, grad_clips, inp_clips, epsilons):

    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=BATCH_SIZE)
    dp_train_loader = DPDataLoader.from_data_loader(train_loader)
    
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=BATCH_SIZE)
    model, info = uc.run_experiment(Classifier(), 
                                    dp_train_loader,
                                    target_eps,
                                    epochs,
                                    inp_clip,
                                    grad_clip,
                                    BATCH_SIZE,
                                    test_loader
                                    )
    tl, correct, set_len = uc.test(model, test_loader)

    print(f'MNIST_d_{BATCH_SIZE}_{epochs}_{grad_clip}_{inp_clip}_{target_eps}', correct/set_len)
    pickle.dump((info, tl, correct), open(f'results/MNIST_d_{BATCH_SIZE}_{epochs}_{grad_clip}_{inp_clip}_{target_eps}.p', 'wb'))
                
