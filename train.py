import seaborn as sns
import torch
import numpy as np
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from dataset import Passenger, scaler
from model import LSTM

df = sns.load_dataset("flights")
# df = pd.read_csv('temperature.csv', header = 0)
dataset = Passenger(df)

train_len = int(len(dataset) * 0.7)
test_len = len(dataset) - train_len
generator = torch.Generator().manual_seed(0)
train_data, test_data = random_split(dataset, [train_len, test_len])

train_loader = DataLoader(train_data, shuffle = False, batch_size = 32, drop_last = True)
test_loader = DataLoader(test_data, shuffle = False, batch_size = 32, drop_last = True)

device = torch.device('cpu')
model = LSTM()
# model = model.to(device)

loss_f = nn.MSELoss()
opt = optim.Adam(model.parameters(), lr = 1e-3)

def train():
    model.train()
    train_loss = 0

    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        opt.zero_grad()
        pred = model(data)
        # pred = pred.view(-1)
        loss = loss_f(pred, target)

        loss.backward()
        opt.step()
        train_loss += loss.item()

    return train_loss / len(train_loader)

def test():
    model.eval()
    test_loss = 0

    for idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        
        opt.zero_grad()
        pred = model(data)
        # pred = pred.view(-1)
        loss = loss_f(pred, target)

        loss.backward()
        opt.step()
        test_loss += loss.item()

    return test_loss / len(test_loader)


if __name__ == "__main__":

    train_losses = []
    test_losses = []
    preds = []

    for i in range(20):
        train_loss = train()
        test_loss = test()

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print("epoch:{}, train_loss:{:.6f}, test_loss{:.6f}".format(i+1, train_loss, test_loss))

    torch.save(model.state_dict(), 'lstm.pt')
