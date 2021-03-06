import os
import numpy as np
import xarray as xr
import random
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

loss_tracker = []
valid_score = []
train_score = []

import torch
import torch.nn as nn

def RMSE(y_true, y_pred):
    return torch.sqrt(torch.mean(torch.square(y_true - y_pred)))

class ScoreLoss(nn.Module):

    def forward(self, y_preds, y_true):
        accskill_score = 0
        rmse_scores = 0
        a = [1.5] * 4 + [2] * 7 + [3] * 7 + [4] * 6
        y_true_mean = torch.mean(y_true, dim=0)
        y_pred_mean = torch.mean(y_preds, dim=0)
        #     print(y_true_mean.shape, y_pred_mean.shape)

        for i in range(24):
            fenzi = torch.sum((y_true[:, i] - y_true_mean[i]) * (y_preds[:, i] - y_pred_mean[i]))
            fenmu = torch.sqrt(torch.sum((y_true[:, i] - y_true_mean[i]) ** 2) * torch.sum(
                (y_preds[:, i] - y_pred_mean[i]) ** 2))
            cor_i = fenzi / fenmu

            accskill_score += a[i] * np.log(i + 1) * cor_i
            rmse_score = RMSE(y_true[:, i], y_preds[:, i])
            #         print(cor_i,  2 / 3.0 * a[i] * np.log(i+1) * cor_i - rmse_score)
            rmse_scores += rmse_score

        return -(2 / 3.0 * accskill_score - rmse_scores)


# python ref/stnn_train.py
def set_seed(seed=427):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)


class EarthDataSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['sst'])

    def __getitem__(self, idx):
        return (self.data['sst'][idx], self.data['t300'][idx], self.data['ua'][idx], self.data['va'][idx]), \
               self.data['label'][idx]


def load_data2():
    # CMIP data
    train = xr.open_dataset('D:/chromeFiles/enso_round1_train_20210201/CMIP_train.nc')
    label = xr.open_dataset('D:/chromeFiles/enso_round1_train_20210201/CMIP_label.nc')

    # feature 前12个月
    train_sst = train['sst'][:, :12].values  #(4645-样本量, 12-12个月, 24-经度, 72-纬度)
    train_t300 = train['t300'][:, :12].values
    train_ua = train['ua'][:, :12].values
    train_va = train['va'][:, :12].values
    # label 后24个月
    train_label = label['nino'][:, 12:36].values # y

    train_ua = np.nan_to_num(train_ua) # Replace NaN with zero and infinity with large finite numbers
    train_va = np.nan_to_num(train_va)
    train_t300 = np.nan_to_num(train_t300)
    train_sst = np.nan_to_num(train_sst)

    # SODA data
    train2 = xr.open_dataset('D:/chromeFiles/enso_round1_train_20210201/SODA_train.nc')
    label2 = xr.open_dataset('D:/chromeFiles/enso_round1_train_20210201/SODA_label.nc')

    train_sst2 = train2['sst'][:, :12].values  # (100, 12, 24, 72)
    train_t3002 = train2['t300'][:, :12].values
    train_ua2 = train2['ua'][:, :12].values
    train_va2 = train2['va'][:, :12].values
    train_label2 = label2['nino'][:, 12:36].values

    print('Train samples: {}, Valid samples: {}'.format(len(train_label), len(train_label2)))

    dict_train = {
        'sst': train_sst,
        't300': train_t300,
        'ua': train_ua,
        'va': train_va,
        'label': train_label}
    dict_valid = {
        'sst': train_sst2,
        't300': train_t3002,
        'ua': train_ua2,
        'va': train_va2,
        'label': train_label2}
    train_dataset = EarthDataSet(dict_train)
    valid_dataset = EarthDataSet(dict_valid)
    return train_dataset, valid_dataset


class EarthDataSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['sst'])

    def __getitem__(self, idx):
        return (self.data['sst'][idx], self.data['t300'][idx], self.data['ua'][idx], self.data['va'][idx]), \
               self.data['label'][idx]


class simpleSpatialTimeNN(nn.Module):
    def __init__(self, n_cnn_layer: int = 1, kernals: list = [3], n_lstm_units: int = 64):
        super(simpleSpatialTimeNN, self).__init__()
        self.conv1 = nn.ModuleList([nn.Conv2d(in_channels=12, out_channels=12, kernel_size=i) for i in kernals])
        self.conv2 = nn.ModuleList([nn.Conv2d(in_channels=12, out_channels=12, kernel_size=i) for i in kernals])
        self.conv3 = nn.ModuleList([nn.Conv2d(in_channels=12, out_channels=12, kernel_size=i) for i in kernals])
        self.conv4 = nn.ModuleList([nn.Conv2d(in_channels=12, out_channels=12, kernel_size=i) for i in kernals])
        self.pool1 = nn.AdaptiveAvgPool2d((22, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((1, 70))
        self.pool3 = nn.AdaptiveAvgPool2d((1, 128))
        self.batch_norm = nn.BatchNorm1d(12, affine=False)
        self.lstm = nn.LSTM(1540 * 4, n_lstm_units, 2, bidirectional=True, batch_first=True)
        # todo lstminput of shape (seq_len, batch, input_size) seq_len = lat*lon = 1540
        # input_size: The number of expected features in the input `x`
        # hidden_size: The number of features in the hidden state `h`
        # num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
        #     would mean stacking two LSTMs together to form a `stacked LSTM`,
        #     with the second LSTM taking in outputs of the first LSTM and
        #     computing the final results. Default: 1
        # fixme the input shape of lstm seems to be mistaken, should b 12, 64, 6160
        #         - **input** of shape `(seq_len, batch, input_size)`: tensor containing the features
        self.linear = nn.Linear(128, 24)

    def forward(self, sst, t300, ua, va):
        for conv1 in self.conv1:
            sst = conv1(sst)  # batch * 12 * (24 - 2) * (72 -2)
        for conv2 in self.conv2:
            t300 = conv2(t300)
        for conv3 in self.conv3:
            ua = conv3(ua)
        for conv4 in self.conv4:
            va = conv4(va)

        sst = torch.flatten(sst, start_dim=2)  # batch * 12 * 1540
        t300 = torch.flatten(t300, start_dim=2)
        ua = torch.flatten(ua, start_dim=2)
        va = torch.flatten(va, start_dim=2)  # if flat, lstm input_dims = 1540 * 4

        x = torch.cat([sst, t300, ua, va], dim=-1)
        x = self.batch_norm(x)
        x, _ = self.lstm(x)

        x = self.pool3(x).squeeze(dim=-2)
        x = self.linear(x)
        return x


def load_model(model_dir):
    print(model_dir)
    model = simpleSpatialTimeNN()
    model.load_state_dict(torch.load(model_dir))
    model.eval()
    return model


def coreff(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    c1 = sum((x - x_mean) * (y - y_mean))
    c2 = sum((x - x_mean) ** 2) * sum((y - y_mean) ** 2)
    return c1 / np.sqrt(c2)


def rmse(preds, y):
    return np.sqrt(sum((preds - y) ** 2) / preds.shape[0])


def eval_score(preds, label):
    # preds = preds.cpu().detach().numpy().squeeze()
    # label = label.cpu().detach().numpy().squeeze()
    acskill = 0
    RMSE = 0
    a = 0
    a = [1.5] * 4 + [2] * 7 + [3] * 7 + [4] * 6
    for i in range(24):
        RMSE += rmse(label[:, i], preds[:, i])
        cor = coreff(label[:, i], preds[:, i])

        acskill += a[i] * np.log(i + 1) * cor
    return 2 / 3 * acskill - RMSE


fit_params = {
    'n_epochs': 7,
    # 'n_epochs': 2,
    'learning_rate': 5e-4,
    'batch_size': 128,
    # 'loss':nn.MSELoss(),
    'loss':ScoreLoss()
}


def train():
    set_seed()
    train_dataset, valid_dataset = load_data2()
    train_loader = DataLoader(train_dataset, batch_size=fit_params['batch_size'])
    valid_loader = DataLoader(valid_dataset, batch_size=fit_params['batch_size'])

    model = simpleSpatialTimeNN()
    # device = 'cpu'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=fit_params['learning_rate'])
    loss_fn = fit_params['loss']
    # loss_fn = ScoreLoss()
    model.to(device)
    loss_fn.to(device)

    for i in range(fit_params['n_epochs']):
        model.train()
        for step, ((sst, t300, ua, va), label) in enumerate(train_loader):
            sst = sst.to(device).float()
            # 这行代码的意思是将tensor变量copy一份到device所指定的GPU/CPU上去
            # 之后的运算都在device所指定的GPU/CPU上进行。
            t300 = t300.to(device).float()
            ua = ua.to(device).float()
            va = va.to(device).float()
            optimizer.zero_grad() # set the gradient back to 0
            label = label.to(device).float()

            preds = model(sst[:1,:,:,:], t300[:1,:,:,:], ua[:1,:,:,:], va[:1,:,:,:])

            preds = model(sst, t300, ua, va)
            loss = loss_fn(preds, label)
            score = eval_score(label.cpu().detach().numpy(), preds.cpu().detach().numpy())
            loss.backward()
            # for shentingwei
            # loss.backward() computes dloss/dx for every parameter x which has requires_grad=True.
            # These are accumulated into x.grad for every parameter x
            # x.grad += dloss/dx
            optimizer.step()
            # optimizer.step updates the value of x using the gradient x.grad. For example, the SGD optimizer performs:
            # x += -lr * x.grad
            loss_tracker.append(loss)
            train_score.append(score)
            # print('Step: {}, Train Loss(score from the coder): {}'.format(step, score))
            print('Step: {}, Train Loss: {}'.format(step, loss))

        model.eval()
        y_true, y_pred = [], []
        for step, ((sst, t300, ua, va), label) in enumerate(valid_loader):
            sst = sst.to(device).float()
            t300 = t300.to(device).float()
            ua = ua.to(device).float()
            va = va.to(device).float()
            label = label.to(device).float()
            preds = model(sst, t300, ua, va)

            y_pred.append(preds)
            y_true.append(label)

        y_true = torch.cat(y_true, axis=0)
        y_pred = torch.cat(y_pred, axis=0)
        sco = eval_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
        for j in range(len(train_loader)):
            valid_score.append(sco)
        print('Epoch: {}, Valid Score {}'.format(i + 1, sco))

    torch.save(model.state_dict(), '../user_data/stnn.pt') # save model
    print('Model saved successfully')


if __name__ == "__main__":
    train()
    plt.figure(1)
    ax = plt.subplot(311)
    ax.set_title('train_loss')
    plt.plot(loss_tracker)
    # plt.show()
    ax1 = plt.subplot(312)
    ax1.set_title('valid_score')
    plt.plot(valid_score)
    # plt.show()
    ax2 = plt.subplot(313)
    ax2.set_title('train_score')
    plt.plot(train_score)
    plt.show()
    print("程序运行完毕")
