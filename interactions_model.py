import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
from torch import optim
import CLR as CLR
import OneCycle as OneCycle
from tqdm import tqdm_notebook as tqdm
from time import time


# used in fastai
def get_emb(ni,nf):
    e = nn.Embedding(ni, nf)
    e.weight.data.uniform_(-0.01,0.01)
    return e


class EmbeddingNet(nn.Module):
    def __init__(self, n_factors, n_users, n_items, min_score, max_score):
        super().__init__()
        self.min_score, self.max_score = min_score, max_score

        # get user and item embeddings
        (self.u, self.i) = [get_emb(*o) for o in [
            (n_users, n_factors), (n_items, n_factors)]]
        self.lin1 = nn.Linear(n_factors*2, 10)  # 10 hidden neurons
        self.lin2 = nn.Linear(10, 1)
        self.drop1 = nn.Dropout(0.05)  # dropout rate of 5%
        self.drop2 = nn.Dropout(0.15)  # dropout rate of 15%

    def forward(self, users, items):
        # concatenate embeddings to form first layer, add dropout
        x = self.drop1(torch.cat([self.u(users),self.i(items)], dim=1))
        # second layer with 10 hidden neurons and dropout
        x = self.drop2(F.relu(self.lin1(x)))
        # output layer with one neuron
        x = self.lin2(x)
        # add sigmoid activation function, but squeeze between min and max score
        return torch.sigmoid(x) * (self.max_score - self.min_score) + self.min_score


def update_lr(optimizer, lr):
    """Update the learning rate of a PyTorch optimizer"""
    for g in optimizer.param_groups:
        g['lr'] = lr


def update_mom(optimizer, mom):
    """Update the momentum of a PyTorch optimizer"""
    for g in optimizer.param_groups:
        g['momentum'] = mom


def find_lr(model, data_loader, optimizer, criterion):
    """Find the optimal learning rate to use for a particular combination of
       model, dataset and optimizer"""
    ## ToDo: include running loss in progress bar

    # t = tqdm(data_loader, leave=False, total=len(data_loader))
    running_loss = 0.
    avg_beta = 0.98
    counter = 0
    clr = CLR.CLR(optimizer, len(data_loader))
    model.train()

    # loop through data until gradient starts to explode
    for user, item, rating in tqdm(data_loader):

        output = model(user, item)
        loss = criterion(output, rating)

        running_loss = avg_beta * running_loss + (1-avg_beta) * loss.item()
        smoothed_loss = running_loss / (1 - avg_beta**(counter+1))
        # t.set_postfix(loss=smoothed_loss)

        lr = clr.calc_lr(smoothed_loss)
        if lr == -1 :
            break
        update_lr(optimizer, lr)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        counter += 1

    # plot learning rate vs loss
    clr.plot()


def fit_model(epochs, model, optimizer, criterion, train, test, one_cycle=None):
    """Train and validate model"""

    train_losses, test_losses = [], []
    for e in range(epochs):

        # training loop
        model.train()
        running_loss = 0.0

        for user, item, rating in tqdm(train):

            optimizer.zero_grad()

            # if using cyclic learning rate
            if one_cycle:
                # calculate new learning rate and momentum
                lr, mom = one_cycle.calc()
                # update learning rate
                update_lr(optimizer, lr)
                # update momentum
                update_mom(optimizer, mom)

            output = model(user, item)
            loss = criterion(output, rating)

            # update weights
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # validation at end of epoch
        else:
            # print training loss
            print("Training Loss: {:.3f}.. ".format(running_loss/len(train)))
            running_loss = 0.0

            # calculate test loss
            test_loss = 0.0

            # turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                # turn off dropout
                model.eval()
                for user, item, rating in test:
                    output = model(user, item)
                    test_loss += criterion(output, rating)

            # print test loss
            print("Test Loss: {:.3f}.. ".format(test_loss/len(test)),
                  "Epoch: {}/{}.. ".format(e+1, epochs))
