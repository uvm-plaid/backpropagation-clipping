
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import time


def accuracy(model, X, y):
    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(y).long()
    outputs = model(Xt)
    y_hat = [1 if o>.5 else 0 for o in outputs]
    accuracy = np.sum(y_hat == y) / len(y)
    return accuracy

def l2_clip(t, C):
    dims = tuple(range(1, len(t.shape)))
    norm = t.norm(dim=dims, keepdim=True, p=2).expand(t.shape)
    clipped = torch.where(norm > C, C*(t/norm), t)

    return clipped

def l1_clip(t, C):
    dims = tuple(range(1, len(t.shape)))
    norm = t.norm(dim=dims, keepdim=True, p=1).expand(t.shape)
    clipped = torch.where(norm > C, C*(t/norm), t)

    return clipped

def lweird_clip(t, C):
    dims = tuple(range(2, len(t.shape)))
    first = t.norm(dim=dims, keepdim=True, p=1)
    norm = first.norm(dim=[1], keepdim=True, p=2).expand(t.shape)
    clipped = torch.where(norm > C, C*(t/norm), t)

    return clipped



def get_clamp_func(norm, bound):

    def clamp_grad_l2(self, grad_input, grad_output):
        g = grad_input[0]
        self.grad_maxes.append(g.abs().max().item())
        return (l2_clip(g, bound),)

    def clamp_grad_lweird(self, grad_input, grad_output):
        g = grad_input[0]
        self.grad_maxes.append(g.abs().max().item())
        return (lweird_clip(g, bound),)

    def clamp_grad_l1(self, grad_input, grad_output):
        g = grad_input[0]
        self.grad_maxes.append(g.abs().max().item())
        return (l1_clip(g, bound),)

    def clamp_input(self, input):
        self.input_maxes.append(input[0].abs().max().item())
        return tuple([l2_clip(x, bound) for x in input])
    
    if norm == 'l2':
        return clamp_grad_l2
    
    if norm == 'l1':
        return clamp_grad_l1
    
    if norm == 'lweird':
        return clamp_grad_lweird
    
    if norm == 'input':
        return clamp_input
    
    

def zcdp_eps(rho, delta):
    return rho + 2*np.sqrt(rho*np.log(1/delta))


def run_experiment(model, train_loader, rho_i, epochs, input_bound, grad_bound):
    model.to('cuda')
    #model.network.to('cuda')
    model_criterion = nn.NLLLoss()
    model_optimizer = optim.Adam(model.parameters(), lr=0.01)#, weight_decay=0.0001)
    total_rho = 0
    
    clamp_grad_lweird = get_clamp_func('lweird', grad_bound)
    clamp_grad_l2 = get_clamp_func('l2', grad_bound)
    clamp_input = get_clamp_func('input', input_bound)
    
    for x in model.l1_clip:
        x.register_backward_hook(clamp_grad_lweird)
    for x in model.l2_clip:
        x.register_backward_hook(clamp_grad_l2)
    for x in model.input_clip:
        x.register_forward_pre_hook(clamp_input)
    
    for x in model.network:
        x.input_maxes = []
        x.grad_maxes = []
#         x.register_backward_hook(clamp_grad)
#         x.register_forward_pre_hook(clamp_input)
    
    sensitivities = []
    norms = []
    decays = []
    losses = []
    
    model.train()
    # sensitivity for everything with weights is just:
    sensitivity = input_bound * grad_bound
    sigma = np.sqrt(sensitivity**2 / (2*rho_i))
    print('sensitivity:', sensitivity)
    
    for epoch in range(epochs):
        for x_batch_train, y_batch_train in train_loader:
            xb = x_batch_train.to('cuda')
            yb = y_batch_train.to('cuda')
            model_optimizer.zero_grad()
            outputs = model.forward(xb)
            loss = model_criterion(outputs, yb)
            losses.append(loss)
            loss.backward()
            
            for p in model.parameters():
                with torch.no_grad():
                    p.grad += sigma * torch.randn(p.shape).to('cuda')

            norms.append(next(model.parameters()).data.norm())

            model_optimizer.step()
    
            

    total_weights = 0
    for p in model.parameters():
        total_rho += rho_i
        total_weights += p.flatten().shape[0]

    total_rho *= epochs

    info = {'sens': sensitivities,
            'norms': norms,
            'decays': decays,
            'losses': losses,
            'total rho': total_rho,
            'total epsilon':zcdp_eps(total_rho, 1e-5),
            'total weights': total_weights}

    return model, info


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to('cuda'), target.to('cuda')
            output = model(data)
            test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    return test_loss, correct, len(test_loader.dataset)
