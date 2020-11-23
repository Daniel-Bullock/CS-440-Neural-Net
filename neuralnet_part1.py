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
within this file and neuralnet_part2 -- the unrevised staff files will be used for all other
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

        For Part 1 the network should have the following architecture (in terms of hidden units):

        in_size -> 32 ->  out_size
        We recommend setting the lrate to 0.01 for part 1

        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.fc1 = nn.Linear(in_size, 32)
        self.fc2 = nn.Linear(32, out_size)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lrate)



    def forward(self, x):
        """ A forward pass of your neural net (evaluates f(x)).

        @param x: an (N, in_size) torch tensor

        @return y: an (N, out_size) torch tensor of output from the network
        """
        temp = x
        m = torch.mean(temp)
        std = torch.std(temp)
        temp = (temp - m) / std
        temp = F.relu(self.fc1(temp))
        temp = self.fc2(temp)
        return temp


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
    """
    '''
    loss_fn = nn.CrossEntropyLoss()
    net = NeuralNet(0.01, loss_fn, len(train_set[0]), 2)
    losses = []

    train_set = (train_set - train_set.mean()) / train_set.std()
    dev_set = (dev_set - dev_set.mean()) / dev_set.std()

    for n in range(n_iter):
        inputs, labels = train_set[(n % 75) * batch_size:(n % 75 + 1) * batch_size], train_labels[
                                                                                     (n % 75) * batch_size:(
                                                                                                                       n % 75 + 1) * batch_size]
        losses.append(net.step(inputs, labels).item())

    yhats = [output.max(dim=0)[1] for output in net.forward(dev_set)]

    return losses, yhats, net
    '''

    nn_loss = nn.CrossEntropyLoss()
    net = NeuralNet(0.01, nn_loss, len(train_set[0]), 2)
    losses = []
    for iteration in range(n_iter):

        loss = 0.0
        for i in range(int(len(train_set) / batch_size) - 1):
            setout = train_set[batch_size * i: batch_size * (i + 1)]
            labelout = train_labels[batch_size * i: batch_size * (i + 1)]

            net.optimizer.zero_grad()
            outputs = net(setout)
            loss = net.loss_fn(outputs, labelout)
            loss.backward()
            net.optimizer.step()
            loss += loss.item()
        losses.append(loss)
    preds = net(dev_set)
    predictions = np.empty(len(preds))

    for i in range(len(preds)):
        if preds[i][0] > preds[i][1]:
            if preds[i][0] < 0.5:
                predictions[i] = 0
            else:
                predictions[i] = 1
        else:
            if preds[i][1] < 0.5:
                predictions[i] = 0
            else:
                predictions[i] = 1

    return losses, predictions, net
