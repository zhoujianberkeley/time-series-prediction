def batch_gen(seed, last_seed):
    """
    总训练集70个SODA,4645*0.7=3252个CMIP;总验证集30个SODA,4645*0.3=1393个CMIP,共1423个
    :param seed:随机数种子;
    :param last_seed: 上次的随机数种子，用于得到last_CMIP_ind;
    :return: 全部SODA，130个上次学过的随机的CMIP,400个另外的随机的CMIP（可能有学过的）;
    训练集600个样本，验证集1423个样本
    """
    with open('pkl/batch_seed_'+str(last_seed)+'.pkl','rb') as pf0:
        _, last_CMIP_ind = pickle.load(pf0)
    del _
    gc.collect()

    np.random.seed(seed)
    CMIP_ind = np.arange(70, 3252)
    SODA_ind = np.arange(70)
    last_CMIP_ind = np.random.choice(last_CMIP_ind, 130)
    left_CMIP_ind = np.array(list(set(CMIP_ind).difference(set(last_CMIP_ind))))
    new_CMIP_ind = np.random.choice(left_CMIP_ind, 400)
    ind = np.hstack((SODA_ind, last_CMIP_ind, new_CMIP_ind))
    with open('pkl/CMIP_scaled_13.pkl', 'rb') as pf1:
        data = pickle.load(pf1)
    data = {'x_train': data['x_train'][ind],
            'x_val': data['x_val'],
            't_train': data['t_train'][ind],
            't_val': data['t_val']}
    with open('pkl/batch_seed_' + str(seed) + '.pkl', 'wb') as pf2:
        pickle.dump((data, new_CMIP_ind), pf2)

#%%
import pickle
from netCDF4 import Dataset
import numpy as np
import gc
import torch

# %%
import pickle
from netCDF4 import Dataset
import numpy as np
import gc
import torch
# %%

with open('pkl/population_stats.pkl', 'rb') as pf0:
    pop_mean, pop_std = pickle.load(pf0)
file_path = 'C:\\Users\\13372\\Downloads\\Compressed\\enso_round1_train_20210201\\'
SODA_sample_path = file_path + 'SODA_train.nc'
CMIP_sample_path = file_path + 'CMIP_train.nc'
SODA_label_path = file_path + 'SODA_label.nc'
CMIP_label_path = file_path + 'CMIP_label.nc'
random_seed = 623


def train_val_split(sample, label, val_size, seed=random_seed):
    """
    :param sample: tuple,(year,12,24,72,4)
    :param label: tuple,(year,24)
    :param val_size: fraction of validation set, between 0 and 1
    :param seed: random seed
    """
    assert sample.ndim == 5 and label.ndim == 2 and val_size <= 1
    np.random.seed(seed)
    sample_num = sample.shape[0]
    shuffle_idx = np.arange(sample_num)
    np.random.shuffle(shuffle_idx)
    val_size = int(val_size * sample_num)
    x_train, x_val = sample[shuffle_idx[:-val_size]], \
                     sample[shuffle_idx[-val_size:]]
    t_train, t_val = label[shuffle_idx[:-val_size]], \
                     label[shuffle_idx[-val_size:]]
    ts = lambda x: torch.from_numpy(x).type(torch.float32)
    return {'x_train': ts(x_train), 'x_val': ts(x_val),
            't_train': ts(t_train), 't_val': ts(t_val)}


def sample_extract(sample_name, batch_id=None):
    extract = lambda x: np.expand_dims(np.array(x[:, :12, :, :]), axis=4)
    assert sample_name in ['SODA', 'CMIP', 'CMIP_batch']
    if sample_name == 'SODA':
        sample_path = SODA_sample_path
    else:
        sample_path = CMIP_sample_path
        if sample_name == 'CMIP_batch':
            extract = lambda x: np.expand_dims(np.array(x[batch_id, :12, :, :]), axis=4)
    sample_nc = Dataset(sample_path, 'r')
    sst = extract(sample_nc.variables['sst'])
    t300 = extract(sample_nc.variables['t300'])
    ua = extract(sample_nc.variables['ua'])
    va = extract(sample_nc.variables['va'])
    sample = np.concatenate((sst, t300, ua, va), axis=-1)
    return sample


def population_stats():
    population = np.concatenate((sample_extract('SODA'), sample_extract('CMIP')), axis=0)
    mean = np.nanmean(population.reshape(-1, 4), axis=0).reshape((1, 1, 1, 1, 4))
    std = np.nanstd(population.reshape(-1, 4), axis=0).reshape((1, 1, 1, 1, 4))
    with open('pkl/population_stats.pkl', 'wb') as pf1:
        pickle.dump((mean, std), pf1)


def nan_fill(data):
    nan = np.isnan(data)
    if nan.any():
        mean = pop_mean.ravel()
        for ft in range(4):
            data[nan[:, :, :, :, ft], ft] = mean[ft]
    return data


def SODA_gen():
    SODA_sample = sample_extract('SODA')
    SODA_sample = nan_fill(SODA_sample)
    SODA_sample = (SODA_sample - pop_mean) / pop_std

    SODA_label_nc = Dataset(SODA_label_path, 'r')
    SODA_label = (np.array(SODA_label_nc.variables['nino'])[:, 12:])
    assert not np.isnan(SODA_label).any()  # label无缺失值
    SODA_data = train_val_split(SODA_sample, SODA_label, val_size=0.3)
    del SODA_label, SODA_label_nc, SODA_sample
    gc.collect()

    with open('pkl/SODA_scaled.pkl', 'wb') as pf2:
        pickle.dump(SODA_data, pf2)

    del SODA_data
    gc.collect()


def CMIP_gen(i):
    total_idx = np.arange(4645)
    np.random.shuffle(total_idx)
    assert 1 <= i <= 13
    if i == 12:
        batch_id = total_idx[4400:4645]
    elif i == 13:
        batch_id = total_idx
    else:
        batch_id = total_idx[400 * (i - 1):400 * i]
    CMIP_sample = sample_extract('CMIP_batch', batch_id=batch_id)
    CMIP_sample = nan_fill(CMIP_sample)
    CMIP_sample = (CMIP_sample - pop_mean) / pop_std

    CMIP_label_nc = Dataset(CMIP_label_path, 'r')
    CMIP_label = (np.array(CMIP_label_nc.variables['nino'])[batch_id, 12:])

    CMIP_data = train_val_split(CMIP_sample, CMIP_label, val_size=0.3)
    del CMIP_label, CMIP_label_nc, CMIP_sample
    gc.collect()

    if i == 13:
        with open('pkl/SODA_scaled.pkl', 'rb') as pf1:
            SODA_data = pickle.load(pf1)
        for key, val in SODA_data.items():
            CMIP_data[key] = torch.cat((val, CMIP_data[key]), dim=0)

    with open('pkl/CMIP_scaled_' + str(i) + '.pkl', 'wb') as pf2:
        pickle.dump(CMIP_data, pf2)
    del CMIP_data
    gc.collect()

# %%
# population_stats()
CMIP_gen(13)

# registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:1.6-cuda10.1-py3