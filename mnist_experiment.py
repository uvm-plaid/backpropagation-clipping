
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import pickle


import time
from torchvision import datasets, transforms

import upstream_clipping as uc


transform=transforms.Compose([
    transforms.ToTensor(),
    #transforms.Lambda(lambda x: torch.flatten(x)),
    #transforms.Normalize((0.1307,), (0.3081,))
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
            torch.nn.Conv2d(1, 16, 8, 2, padding=3, bias=False),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 1), 
            torch.nn.Conv2d(16, 32, 4, 2, bias=False),
            torch.nn.ReLU(), 
            torch.nn.MaxPool2d(2, 1), 
            torch.nn.Flatten(), 
            torch.nn.Linear(32 * 4 * 4, 32, bias=False),
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
    


for BATCH_SIZE in [512, 1024, 2048, 4096]:
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=BATCH_SIZE)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=BATCH_SIZE)
    for epochs in [15, 25, 40, 100]:
        for grad_clip in [0.00001, 0.0001, 0.001, 0.01, 0.1]:
            for inp_clip in [0.1, 1, 5, 10]:
                for rho_i in [0.00001, 0.00005, 0.0001]:
                    model, info = uc.run_experiment(Classifier(), 
                                                train_loader,
                                                rho_i,
                                                epochs,
                                                inp_clip,
                                                grad_clip
                                            )
                    tl, correct, set_len = uc.test(model, test_loader)
                    print(f'MNIST_{BATCH_SIZE}_{epochs}_{grad_clip}_{inp_clip}_{rho_i}', correct/set_len)
                    pickle.dump((info, tl, correct), open(f'MNIST_{BATCH_SIZE}_{epochs}_{grad_clip}_{inp_clip}_{rho_i}.p', 'wb'))
                
