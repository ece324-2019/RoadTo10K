import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
from model import MultiLayerPerceptron
from dataset import AdultDataset
from Baselinemodel import LogisticRegressionModel
import matplotlib.pyplot as plt



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
torch.manual_seed(100)
processed_data = pd.DataFrame(picks_onehot)
processed_data.to_csv('processed_test.csv')

train_data, validation_data, train_label, valid_label = train_test_split(picks_onehot, radiant_win_onehot,
                                                                         test_size=0.2, random_state=0)
train_set = AdultDataset(train_data, train_label)
train_loader = DataLoader(train_set, batch_size=20000, shuffle=True)
val_set = AdultDataset(validation_data, valid_label)
val_loader = DataLoader(val_set, batch_size=4000, shuffle=False)

model = LogisticRegressionModel(258)
loss_fnc = torch.nn.CrossEntropyLoss()
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs=50
training_loss_epoch_array=[]
training_accuracy = []
valid_loss_epoch_array = []
valid_accuracy = []
numepoch=[]
num=0


def evaluate(model, loader):
    total_corr = 0
    total = 0
    for x, vbatch in enumerate(loader):
        data, labels = vbatch
        prediction = model(data.float())
        _,prediction_results = torch.max(prediction, 1)
        total_corr += (prediction_results == labels).sum()
        total += labels.size(0)

    return total_corr.item()/total

for epoch in range(0, epochs):
        loss_epoch = 0
        total_corr = 0
        total = 0

        for i, batch in enumerate(train_loader, 0):
            data, labels = batch
            optimizer.zero_grad()
            prediction = model(data.float())
            loss = loss_fnc(prediction, labels)
            loss.backward()
            optimizer.step()
        trainresult = evaluate(model, train_loader)
        validresult = evaluate(model,val_loader)
        #loss_record.append(loss_epoch)
        training_accuracy .append(trainresult)
        valid_accuracy.append(validresult)
        num+=1
        print(num)
        numepoch.append(num)
'''
def evaluate(model, iter):
    total_correct = 0
    total = 0
    # total_loss =0
    for j, vbatch in enumerate(iter):
        data, label = vbatch
        optimizer.zero_grad()
        predictions = model(data.float())
        corr = (predictions > 0.5).squeeze().long() == label
        total_correct += int(corr.sum())

        total += label.size(0)

    accuracy = total_correct / total
    return accuracy
for epoch in range(epochs):
    for i, batch in enumerate(train_loader):
        data, label = batch
        optimizer.zero_grad()
        predictions = model(data.float())
        batch_loss = loss_fnc(input=predictions.squeeze(), target=label.float())
        batch_loss.backward()
        optimizer.step()

    trainresult = evaluate(model,train_loader)
    validresult =evaluate(model,val_loader)
    num +=1
    numepoch.append(num)
    #training_loss_epoch_array.append(trainresult[0])
    training_accuracy.append(trainresult)
    #valid_loss_epoch_array.append(validresult[0])
    valid_accuracy.append(validresult)

plt.plot(numepoch, training_loss_epoch_array,label='train')
plt.plot(numepoch, valid_loss_epoch_array,label='valid')
plt.xlabel('number of epoch')
plt.ylabel('loss')
plt.title(' data loss vs epoch')
plt.show()
'''
plt.plot(numepoch, training_accuracy,label='train')
plt.plot(numepoch, valid_accuracy, label='valid')
plt.xlabel('number of epoch')
plt.ylabel('accuracy')
plt.title('Training data accuracy vs epoch')
plt.show()
'''
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--eval_every', type=int, default=10)

    args = parser.parse_args()
if __name__ == "__main__":
    main()
'''