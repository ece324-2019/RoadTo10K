import argparse
from time import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import MultiLayerPerceptron
from dataset import AdultDataset
from util import *

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from scipy.signal import savgol_filter
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
winrateforall =[]

# find if there is hero never be selected
count0 = []
for i in range(len(picks_onehot[0])):
    count=0
    for j in range(len(picks_onehot)):
        count+=picks_onehot[j][i]
    count0.append(count)
for i in range(len(count0)):
    if count0[i]==0:
        print(i)
        #return False

print(count0)
# 23 114 115 116 117 121 122 123 124 125 152 243 244 245 246 250 251 252 253 254 255 256 not picked

#winrate overall
for j in range(129):
    wincount = 0
    pickcount=0
    for i in range(len(picks_onehot)):
        #if it is on dire team
        if picks_onehot[i][j]==1:
            pickcount+=1
            #dire team win
            if radiant_win_onehot[i]==0:
             wincount+=1
        # if it is on radiant team
        if picks_onehot[i][j+129]==1:
            pickcount+=1
            #radiant team win
            if radiant_win_onehot[i]==1:
             wincount+=1
    if pickcount==0:
        winrate='NL'
    else:
        winrate=round(float(wincount/pickcount),4)
    winrateforall.append(winrate)
print(winrateforall)

#each hero's winrate against other heros
winvsrate_for_all=[]
for j in range(129):
    winvsrate_for_onehero=[]

    for k in range(129):
        wincount = 0
        vscount = 0
        for i in range(len(picks_onehot)):
            #if picked hero is on dire team
            if picks_onehot[i][j]==1:
                #so opponents are on radiant team
                if picks_onehot[i][k+129]==1:
                    vscount +=1
                    if radiant_win_onehot[i] == 0:
                        wincount += 1
            # if picked hero is on radiant team
            if picks_onehot[i][j+129]==1:
                #so opponents are on dire team
                if picks_onehot[i][k]==1:
                    vscount +=1
                    if radiant_win_onehot[i] == 0:
                        wincount += 1
        if vscount ==0:
            winrate = 'NL'
        else:
            winrate=round(float(wincount/vscount),4)
        winvsrate_for_onehero.append(winrate)
    winvsrate_for_all.append(winvsrate_for_onehero)
# a dimension 129*129 matrix element winvsrate_for_all and winvsrate_for_all[i] represents the winrate for the ith hero against
#all other 128 heros and winvsrate_for_all[i][j] is the win rate when ith hero facing jth hero. note winvsrate_for_all[i][i]
#means the hero is facing himself which is impossible so the value will be 'NL'
print(len(winvsrate_for_all),len(winvsrate_for_all[0]))
for i in range(len(winvsrate_for_all)):
    print(winvsrate_for_all[i])







