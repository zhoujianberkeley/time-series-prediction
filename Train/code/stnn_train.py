import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import xarray as xr
import random
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

loss_tracker = []
valid_score = []
train_score = []


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
    # CMIP train_data
    train = xr.open_dataset(Path(os.path.abspath(os.pardir), "train_data", "enso_round1_train_20210201/CMIP_train.nc"))
    label = xr.open_dataset(Path(os.path.abspath(os.pardir), "train_data", "enso_round1_train_20210201/CMIP_label.nc"))

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

    # SODA train_data

    train2 = xr.open_dataset(Path(os.path.abspath(os.pardir), "train_data", "enso_round1_train_20210201/SODA_train.nc"))
    label2 = xr.open_dataset(Path(os.path.abspath(os.pardir), "train_data", "enso_round1_train_20210201/SODA_label.nc"))

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


def load_data_mix():
    # CMIP train_data
    train = xr.open_dataset(Path(os.path.abspath(os.pardir), "train_data", "enso_round1_train_20210201/CMIP_train.nc"))
    label = xr.open_dataset(Path(os.path.abspath(os.pardir), "train_data", "enso_round1_train_20210201/CMIP_label.nc"))

    # feature 前12个月
    train_sst = train['sst'][:, :12].values  #(4645-样本量, 12-12个月, 24-经度, 72-纬度)
    train_t300 = train['t300'][:, :12].values
    train_ua = train['ua'][:, :12].values
    train_va = train['va'][:, :12].values
    # label 后24个月
    train_label = label['nino'][:, 12:36].values # y

    # SODA train_data

    train2 = xr.open_dataset(Path(os.path.abspath(os.pardir), "train_data", "enso_round1_train_20210201/SODA_train.nc"))
    label2 = xr.open_dataset(Path(os.path.abspath(os.pardir), "train_data", "enso_round1_train_20210201/SODA_label.nc"))


    num_of_soda = fit_params['num_of_soda_in_a_all_soda_train']

    train_sst2 = train2['sst'][num_of_soda:, :12].values  # (100, 12, 24, 72)
    train_t3002 = train2['t300'][num_of_soda:, :12].values
    train_ua2 = train2['ua'][num_of_soda:, :12].values
    train_va2 = train2['va'][num_of_soda:, :12].values
    train_label2 = label2['nino'][num_of_soda:, 12:36].values

    train_sst_soda = train2['sst'][:num_of_soda, :12].values  # (100, 12, 24, 72)
    train_t300_soda = train2['t300'][:num_of_soda, :12].values
    train_ua_soda = train2['ua'][:num_of_soda, :12].values
    train_va_soda = train2['va'][:num_of_soda, :12].values
    train_label_soda = label2['nino'][:num_of_soda, 12:36].values

    # For generalization purpose,we need to add more soda data to the main
    for i in tqdm(range(fit_params['num_of_all_soda_in_train'])):
        train_sst = np.vstack([train_sst,train_sst_soda])
        train_t300 = np.vstack([train_t300,train_t300_soda])
        train_ua = np.vstack([train_ua,train_ua_soda])
        train_va = np.vstack([train_va,train_va_soda])
        train_label = np.vstack([train_label,train_label_soda])

    # after appending the soda data training data will be about 4600+500

    train_ua = np.nan_to_num(train_ua) # Replace NaN with zero and infinity with large finite numbers
    train_va = np.nan_to_num(train_va)
    train_t300 = np.nan_to_num(train_t300)
    train_sst = np.nan_to_num(train_sst)

    rand_order = np.random.permutation((len(train_sst)))

    train_sst = train_sst[rand_order]
    train_t300 = train_t300[rand_order]
    train_ua = train_ua[rand_order]
    train_va = train_va[rand_order]
    train_label = train_label[rand_order]






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


def train():
    set_seed()
    train_dataset, valid_dataset = load_data_mix()
    train_loader = DataLoader(train_dataset, batch_size=fit_params['batch_size'])
    valid_loader = DataLoader(valid_dataset, batch_size=fit_params['batch_size'])

    from Predict.code.model_structure import simpleSpatialTimeNN
    model = simpleSpatialTimeNN()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("training device : ", device)
    optimizer = torch.optim.Adam(model.parameters(), lr=fit_params['learning_rate'])
    loss_fn = fit_params['loss']
    # loss_fn = ScoreLoss()
    model.to(device)
    loss_fn.to(device)

    tis = time.time()
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

    tic = time.time()
    print("model train time : ", tic - tic, " train device : ", device)

    torch.save(model.state_dict(), Path(Path(__file__).parents[2], "Predict/user_data/stnn.pt")) # save model
    print('Model saved successfully')


fit_params = {
    'n_epochs': 22,
    # 'n_epochs': 2,
    'learning_rate': 4e-3,
    'num_of_soda_in_a_all_soda_train':60,
    'num_of_all_soda_in_train':40,
    'batch_size': 64,
    # 'loss':nn.MSELoss(),
    'loss':ScoreLoss()
}

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
