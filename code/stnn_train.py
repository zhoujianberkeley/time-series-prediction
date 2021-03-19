import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader




# python ref/stnn_train.py
def set_seed(seed=427):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)


def load_data2():
    import xarray as xr
    # CMIP data
    train = xr.open_dataset('../data/enso_round1_train_20210201/CMIP_train.nc')
    label = xr.open_dataset('../data/enso_round1_train_20210201/CMIP_label.nc')

    train_sst = train['sst'][:, :12].values  # (4645, 12, 24, 72)
    train_t300 = train['t300'][:, :12].values
    train_ua = train['ua'][:, :12].values
    train_va = train['va'][:, :12].values
    train_label = label['nino'][:, 12:36].values

    train_ua = np.nan_to_num(train_ua) # Replace NaN with zero and infinity with large finite numbers
    train_va = np.nan_to_num(train_va)
    train_t300 = np.nan_to_num(train_t300)
    train_sst = np.nan_to_num(train_sst)

    # SODA data
    train2 = xr.open_dataset('../data/enso_round1_train_20210201/SODA_train.nc')
    label2 = xr.open_dataset('../data/enso_round1_train_20210201/SODA_label.nc')

    train_sst2 = train2['sst'][:, :12].values  # (4645, 12, 24, 72)
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
    # 'n_epochs': 22,
    'n_epochs': 2,
    'learning_rate': 8e-5,
    'batch_size': 64,
}


def train():
    set_seed()
    train_dataset, valid_dataset = load_data2()
    train_loader = DataLoader(train_dataset, batch_size=fit_params['batch_size'])
    valid_loader = DataLoader(valid_dataset, batch_size=fit_params['batch_size'])

    model = simpleSpatialTimeNN()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optimizer = torch.optim.Adam(model.parameters(), lr=fit_params['learning_rate'])
    loss_fn = nn.MSELoss()

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
            loss.backward()
            # for shentingwei
            # loss.backward() computes dloss/dx for every parameter x which has requires_grad=True.
            # These are accumulated into x.grad for every parameter x
            # x.grad += dloss/dx
            optimizer.step()
            # optimizer.step updates the value of x using the gradient x.grad. For example, the SGD optimizer performs:
            # x += -lr * x.grad
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
        print('Epoch: {}, Valid Score {}'.format(i + 1, sco))

    torch.save(model.state_dict(), '../user_data/stnn.pt')
    # torch.save(model, '../user_data/stnn.pkl')

    print('Model saved successfully')


if __name__ == "__main__":
    train()