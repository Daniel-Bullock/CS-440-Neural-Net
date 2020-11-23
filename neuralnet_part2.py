# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
"""
This is the main entry point for MP6. You should only modify code
within this file and neuralnet_part1 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNet(torch.nn.Module):
    def __init__(self, lrate,loss_fn,in_size,out_size):
        """
        Initialize the layers of your neural network

        @param lrate: The learning rate for the model.
        @param loss_fn: A loss function defined in the following way:
            @param yhat - an (N,out_size) tensor
            @param y - an (N,) tensor
            @return l(x,y) an () tensor that is the mean loss
        @param in_size: Dimension of input
        @param out_size: Dimension of output




        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.in_size = 3135
        self.conv1 = nn.Conv2d(3, 16, 4)
        self.fc1 = nn.Linear(3136, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, out_size)
        self.pool = nn.MaxPool2d(2, 2)

        self.optimizer = torch.optim.SGD(self.parameters(), lr=lrate)



    def forward(self, x):
        """ A forward pass of your neural net (evaluates f(x)).

        @param x: an (N, in_size) torch tensor

        @return y: an (N, out_size) torch tensor of output from the network
        """

        x = (x - torch.mean(x)) / torch.std(x)

        x = x.view(-1, 3, 32, 32)
        x = self.pool(F.relu(self.conv1(x)))

        x = x.view(-1, 3136)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = self.fc3(x)
        return x

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y
        @param x: an (N, in_size) torch tensor
        @param y: an (N,) torch tensor
        @return L: total empirical risk (mean of losses) at this time step as a float
        """
        fwd = self.forward(x)
        output = self.loss_fn(fwd, y)
        self.optimizer.step()

        return output



def fit(train_set,train_labels,dev_set,n_iter,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) torch tensor
    @param train_labels: an (N,) torch tensor
    @param dev_set: an (M,) torch tensor
    @param n_iter: int, the number of iterations of training
    @param batch_size: The size of each batch to train on. (default 100)

    # return all of these:

    @return losses: Array of total loss at the beginning and after each iteration. Ensure len(losses) == n_iter
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: A NeuralNet object

    # NOTE: This must work for arbitrary M and N

    model's performance could be sensitive to the choice of learning_rate. We recommend trying different values in case
    your first choice does not seem to work well.
    """

    entropy_loss = nn.CrossEntropyLoss()
    network = NeuralNet(0.01, entropy_loss, len(train_set[0]), 2)

    dim = 0
    loss_out = []

    for iteration in range(n_iter):
        loss = 0.0
        set_out = train_set[batch_size * dim: batch_size * (dim + 1)]
        label_out = train_labels[batch_size * dim: batch_size * (dim + 1)]

        network.optimizer.zero_grad()
        loss = network.loss_fn(network(set_out), label_out)
        loss.backward()
        network.optimizer.step()
        loss += loss.item()
        loss_out.append(loss)
        dim = (dim + 1) % (int(len(train_set) / batch_size))

    preds = network(dev_set)
    predictions = (torch.argmax(preds, 1)).numpy()

    return loss_out, predictions, network
