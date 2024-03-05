import torch 
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from  torch.utils.data import Dataset

scaler = MinMaxScaler(feature_range = (-1, 1))

class temp(Dataset):
    def __init__(self, data):
        self.df = data
        self.org_data = self.df.mean_temp.to_numpy()
        self.nor_data = np.copy(self.org_data)
        self.nor_data = self.nor_data.reshape(-1, 1)
        self.nor_data = scaler.fit_transform(self.nor_data)
        self.nor_data = self.nor_data.reshape(-1, 1)
        self.sample_len = 12

    def __len__(self):
        if len(self.org_data) > self.sample_len:
            return len(self.org_data) - self.sample_len
        else:
            return 0
        
    def __getitem__(self, index):
        target = self.nor_data[self.sample_len + index]
        target = np.array(target).astype(np.float32)

        input = self.nor_data[index : (index + self.sample_len)]
        input = input.reshape(-1, 1)

        input = torch.from_numpy(input).float()
        target = torch.from_numpy(target).float()

        return input, target
    

if __name__ == "__main__":
    df = pd.read_csv('temperature.csv', header = 0)
    dataset = temp(df)
    print(dataset[0])


    
