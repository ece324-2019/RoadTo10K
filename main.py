import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import MultiLayerPerceptron
from dataset import Dataset

matches = pd.read_csv('data-copy.csv')
data = matches[['dire_team', 'radiant_team','radiant_win']]
#data['radiant_team'] = data['radiant_team'] + 129
dire_picks= data['dire_team'].to_numpy()
radiant_picks = data['radiant_team'].to_numpy()
radiant_win = data['radiant_win'].to_numpy()
picks_onehot = np.zeros((dire_picks.size,258), dtype=int)
radiant_win_onehot = np.zeros(dire_picks.size, dtype=int)
for i in range(0,dire_picks.size):
    dire_picks[i] = np.asarray(dire_picks[i].split(",")).astype(int)
    radiant_picks[i] = np.asarray(radiant_picks[i].split(",")).astype(int)
    if radiant_win[i]:
        radiant_win_onehot[i] = 1
    else:
        radiant_win_onehot[i] = 0
    for j in range(0,5):
        picks_onehot[i][dire_picks[i][j] - 1] = 1
        picks_onehot[i][radiant_picks[i][j] - 1 + 129] = 1

processed_data = pd.DataFrame(picks_onehot)
processed_data.to_csv('processed_test.csv')

train_data, validation_data, train_label, valid_label = train_test_split(picks_onehot, radiant_win_onehot,
                                                                         test_size=0.2, random_state=0)
train_set = Dataset(train_data, train_label)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_set = Dataset(validation_data, valid_label)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

