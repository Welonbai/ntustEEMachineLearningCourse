import torch
import pandas as pd
import numpy as np
from model import LSTM
from dataset import temp, scaler
import matplotlib.pyplot as plt
device = torch.device('cpu')
model = LSTM().to(device)
model.load_state_dict(torch.load("lstm.pt"))

def pred(data):
    model.eval()

    with torch.no_grad():
        data = data.to(device)
        y = model(data)
        return y

df = pd.read_csv('temperature.csv', header = 0)
dataset = temp(df)

preds = []

for i in range(len(dataset)):
    data, target = dataset[i]
    data = data.view(1, 12, 1)
    pred_temp = pred(data)
    pred_temp = pred_temp.cpu().detach().numpy()
    

    act_temp = scaler.inverse_transform(pred_temp.reshape(-1, 1))
    print(act_temp)
    preds.append(act_temp.item())

months = range(0, df.month.size)
mena_temp = df.mean_temp

plt.figure(1)
plt.plot(months, mena_temp, label = "org")
plt.plot(months[12:], preds, label = 'pred')
plt.legend()
plt.show()

