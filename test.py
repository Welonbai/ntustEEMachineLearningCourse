import torch
import pandas as pd
import numpy as np
from model import LSTM
from dataset import temp, scaler
import matplotlib.pyplot as plt
from torch.utils.data.dataset import random_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from train import test_data, train_len


device = torch.device('cpu')
model = LSTM().to(device)
model.load_state_dict(torch.load("pt/lstm40.pt"))

def pred(data):
    model.eval()

    with torch.no_grad():
        data = data.to(device)
        y = model(data)
        return y

df = pd.read_csv('DailyDelhiClimate.csv', header = 0)
dataset = temp(df)
test_len = len(test_data)

preds = []

for i in range(test_len):
    data, target = dataset[train_len + i]
    data = data.view(1, 90, 1)
    pred_temp = pred(data)
    pred_temp = pred_temp.cpu().detach().numpy()
    
    act_temp = scaler.inverse_transform(pred_temp)
    print(act_temp)
    preds.append(act_temp.item())

Date = range(0, test_len)
meantemp = df.meantemp[-len(test_data):]
print(meantemp)

plt.figure(1)
plt.plot(Date, meantemp, label = "org")
plt.plot(Date, preds, label = 'pred')
plt.legend()
plt.show()

mse = mean_squared_error(meantemp, preds)
mae = mean_absolute_error(meantemp, preds)

print(f"MSE: {mse:.2f}")
print(f"MAE: {mae:.2f}")

