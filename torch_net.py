"""
Implementation of fully-connected network in PyTorch

Author: Gabriel Maher
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DataSet:
    def __init__(self, X, Y, batch_size=True, shuffle=True):
        self.data   = torch.utils.data.TensorDataset(torch.tensor(X).float(),torch.tensor(Y).float())
        self.loader = torch.utils.data.DataLoader(self.data, batch_size=batch_size, shuffle=True)

class FCLayer(nn.Module):
    def __init__(self, input_size, output_size, activation, drop_rate):
        super(FCLayer, self).__init__()
        self.input_size  = input_size
        self.output_size = output_size
        self.activation  = activation
        self.drop_rate   = drop_rate

        self.linear = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(self.drop_rate)

    def forward(self,x):
        o = self.linear(x)
        o = self.activation(o)
        return self.dropout(o)

class FCNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes,
    output_activation=nn.Identity(),
    hidden_activation=F.relu,
    drop_rate=0.5):
        super(FCNet,self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.drop_rate = drop_rate

        e_sizes = [input_size]+hidden_sizes
        e_layers = nn.ModuleList()
        for i in range(len(e_sizes)-1):
            l = FCLayer(e_sizes[i], e_sizes[i+1], hidden_activation, self.drop_rate)
            e_layers.append(l)

        l = FCLayer(e_sizes[-1], output_size, output_activation, 0.0)
        e_layers.append(l)

        self.e_layers = e_layers

    def forward(self, x):
        o = x
        for l in self.e_layers:
            o = l(o)
        return o

    def predict(self,x):
        o = torch.from_numpy(x).float()
        o = self.forward(o)
        return o.data.numpy()

    def fit(self,x,y,device=torch.device('cpu'), loss_fn=nn.MSELoss(),
        learning_rate=0.001, epochs=50, batch_size=32, log_interval=10):
        dataset = torch.utils.data.TensorDataset(torch.tensor(x).float(),
            torch.tensor(y).float())

        data_loader = torch.utils.data.DataLoader(dataset,
            batch_size=batch_size, shuffle=True)

        opt = torch.optim.Adam(self.parameters(),lr=learning_rate)

        self.train()
        for E in range(epochs):
            train(self,device,data_loader,loss_fn,opt,E,log_interval)
        self.eval()

def train(model, device, train_loader, loss_fn, optimizer, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
