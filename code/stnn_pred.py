import os
import numpy as np
import zipfile
import shutil
import torch

 # 提交用脚本
def predict_single(data_dir, file, model):
    test_data = np.load(os.path.join(data_dir, file))
    # start_month = int(file.split('_')[2])

    # data = transform2train(data, start_month=start_month)
    # sst, t300, ua, va = prepare_data(data)
    # reshape the testing data to make it match with the trained model

    sst = test_data[:, :, :, 0]
    t300 = test_data[:, :, :, 1]
    ua = test_data[:, :, :, 2]
    va = test_data[:, :, :, 3]
    # print(sst.shape)
    x = [i[np.newaxis, ...].astype(np.float32) for i in [sst, t300, ua, va]]
    # x = tuple([i[np.newaxis, ...].astype(np.float32) for i in [sst, t300, ua, va]])
    sst = torch.from_numpy(x[0])
    t300 = torch.from_numpy(x[1])
    ua = torch.from_numpy(x[2])
    va = torch.from_numpy(x[3])

    y = model(sst, t300, ua, va)
    return y.detach().numpy().reshape(-1)  # + 0.04


def predict(data_dir='../tcdata/enso_round1_test_20210201',
            model_dir='../user_data/fine'):  # 提交时： '../tcdata/enso_round1_test_20210201'
    if os.path.exists('../result'):
        shutil.rmtree('../result', ignore_errors=True)
    os.makedirs('../result')

    from stnn_train import load_model
    model = load_model(model_dir)

    ### 1. 测试数据读取
    files = os.listdir(data_dir)
    for file in files:
        if not (file not in ['.DS_Store']) and not (os.path.isdir(os.path.join(data_dir, file))):
            continue
        if not file.endswith(".npy"):
            continue
        print(file)
        ### 2. 结果预测
        res = predict_single(data_dir, file, model)
        np.save('../result/{}'.format(file), res)
    return

# 选手针对测试集中每个测试样本，预测未来24个月的Nino3.4指数，并保存为和测试样本同名的npy格式文件，
# 其中Nino3.4指数均以float格式保存。例如针对test_00001_01_12_.npy样本，
# 预测结果保存为test_00001_01_12_.npy。将测试集所有样本预测结果文件保存在result文件夹下，并打包为result.zip。


def compress(res_dir='../result', output_dir='../result.zip'):
    z = zipfile.ZipFile(output_dir, 'w')
    for d in os.listdir(res_dir):
        z.write(res_dir + os.sep + d)
    z.close()


def local_test(model_dir):
    # model = tf.saved_model.load('../user_data/stnn')
    from stnn_train import load_model
    model = load_model(model_dir)
    y = predict_single(data_dir='../tcdata/enso_round1_test_20210201', file='test_0144-01-12.npy', model=model)
    y = y + 0.04
    print(y)


if __name__ == '__main__':
    model_dir = '../user_data/stnn.pt'
    predict(model_dir=model_dir)
    compress()
    print("finish")
    # local_test(model_dir)
