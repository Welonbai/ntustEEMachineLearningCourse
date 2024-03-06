import torch 
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

scaler = MinMaxScaler(feature_range = (-1, 1))

class temp(Dataset):
    def __init__(self, data):
        self.df = data
        self.org_data = self.df.meantemp.to_numpy()
        self.nor_data = np.copy(self.org_data)
        self.nor_data = self.nor_data.reshape(-1, 1)
        self.nor_data = scaler.fit_transform(self.nor_data)
        self.nor_data = self.nor_data.reshape(-1, 1)
        self.sample_len = 90

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
    df = pd.read_csv('DailyDelhiClimate.csv', header = 0)
    dataset = temp(df)
    print(dataset[0])


# import torch
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from torch.utils.data import Dataset

# scalers = {}  # Dictionary to store individual scalers for each feature


# class temp(Dataset):
#     def __init__(self, data):
#         self.df = data
#         self.features = ['Open', 'High', 'Low', 'Close', 'Volume']
#         self.num_features = len(self.features)
#         self.data = self.df[self.features].values
#         self.normalized_data = self.normalize_data(self.data)
#         self.sample_len = 30

#     def normalize_data(self, data):
#         normalized_data = np.empty_like(data)
#         for i, feature in enumerate(self.features):
#             scaler = MinMaxScaler(feature_range=(-1, 1))
#             normalized_data[:, i] = scaler.fit_transform(data[:, i].reshape(-1, 1)).flatten()
#             scalers[feature] = scaler
#         return normalized_data

#     def __len__(self):
#         return len(self.normalized_data) - self.sample_len

#     def __getitem__(self, index):
#         target = self.normalized_data[self.sample_len + index, self.features.index('Open')]
#         target = np.array(target).astype(np.float32)

#         input_data = self.normalized_data[index: index + self.sample_len]

#         input_tensor = torch.from_numpy(input_data).float()
#         target_tensor = torch.from_numpy(target).float().unsqueeze(-1)

#         return input_tensor, target_tensor

# if __name__ == "__main__":
#     df = pd.read_csv('KO.csv', header=0)  # Replace 'your_data.csv' with your file name
#     dataset = temp(df)
#     print(dataset[0])

    
