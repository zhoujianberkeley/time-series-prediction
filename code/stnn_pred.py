import sys
import os
import json
import numpy as np
import pandas as pd
import joblib
import zipfile
import shutil
import itertools


def predict_single(data_dir, file, model):
    data = np.load(os.path.join(data_dir, file))
    start_month = int(file.split('_')[2])

    data = transform2train(data, start_month=start_month)
    sst, t300, ua, va = prepare_data(data)

    # print(sst.shape)
    x = tuple([i[np.newaxis, ...].astype(np.float32) for i in [sst, t300, ua, va]])
    y = model(x)
    return y.numpy().reshape(-1)  # + 0.04


def predict(data_dir='../tcdata/enso_round1_test_20210201',
            model_dir='../user_data/fine'):  # 提交时： '../tcdata/enso_round1_test_20210201'
    if os.path.exists('../result'):
        shutil.rmtree('../result', ignore_errors=True)
    os.makedirs('../result')

    model = tf.saved_model.load(model_dir)

    for file in os.listdir(data_dir):
        res = predict_single(data_dir, file, model)
        np.save('../result/{}'.format(file), res)
    return


def compress(res_dir='../result', output_dir='result.zip'):
    z = zipfile.ZipFile(output_dir, 'w')
    for d in os.listdir(res_dir):
        z.write(res_dir + os.sep + d)
    z.close()


def local_test():
    model = tf.saved_model.load('../user_data/stnn')
    y = predict_single(data_dir='../data/enso_round1_test_20210201', file='test_0144_01_12.npy', model=model)
    y = y + 0.04
    print(y)


if __name__ == '__main__':
    model_dir = '../user_data/nn'
    predict(model_dir=model_dir)
    compress()