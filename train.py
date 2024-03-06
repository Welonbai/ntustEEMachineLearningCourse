import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from dataset import temp
from model import LSTM
import matplotlib.pyplot as plt

df = pd.read_csv('DailyDelhiClimate.csv', header = 0)
dataset = temp(df)

train_len = int(len(dataset) * 0.7)
test_len = len(dataset) - train_len
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

    for idx, (data, target) in tqdm(enumerate(train_loader)):
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

    for i in tqdm(range(40)):
        train_loss = train()
        test_loss = test()

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        torch.save(model.state_dict(), f'pt/lstm{i+1}.pt')
        print("epoch:{}, train_loss:{:.6f}, test_loss:{:.6f}".format(i+1, train_loss, test_loss))

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Test Losses')
    plt.legend()
    plt.show()


    
