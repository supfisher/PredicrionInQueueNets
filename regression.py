import argparse
from utils import *
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
import numpy as np
from scipy import sparse
from sklearn.metrics import make_scorer


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=20,
                    help='Number of epochs to train.')
parser.add_argument('--pre_len', type=int, default=5,
                    help='the length of input data sequence.')
parser.add_argument('--tar_len', type=int, default=3,
                    help='the length of output target sequence.')


def preprocess_data(features, targets, args):
    """features:(len(times), num_nodes, node_feature)
    targets: (len(times), target_feature)
    retrun: (data_vev, target_vec)
    data_vev:(features.shape[0]-pre_len-tar_len, num_nodes*node_feature)
    target_vec:(tar_len, node_feature, data_vec.shape[0])
    for simplicity we use node_features[0] as the target
    """
    data_vec = np.empty([features.shape[0] - args.pre_len - args.tar_len, args.pre_len*features.shape[1]*features.shape[2]])
    target_vec = np.empty([args.tar_len, targets.shape[-1], data_vec.shape[0]])

    orders = torch.randperm(features.shape[0] - args.pre_len - args.tar_len)
    for id in orders:
        data_vec[id] = features[id:id+args.pre_len].reshape(-1)
        target_vec[:, :, id] = targets[id+args.pre_len-1:id+args.pre_len+args.tar_len-1, :]

    return sparse.csr_matrix(data_vec), target_vec


def data_init(args):
    adj, features, targets = load_path()
    features_train, features_val, features_test, targets_train, targets_val, targets_test \
        = train_test_split(features, targets, ratio=(0.5, 0.3,))
    features_train, features_val, features_test = np.array(features_train), np.array(
        features_val), np.array(features_test)

    targets_train, targets_val, targets_test = np.array(targets_train), np.array(
        targets_val), np.array(targets_test)
    print("featues_train shape: ", features_train.shape)
    args.num_nodes = features_train.shape[1]
    assert adj.shape[0] == features_train.shape[1]
    args.node_features = features_train.shape[2]

    data_train, targets_train = preprocess_data(features_train, targets_train, args)
    data_val, targets_val = preprocess_data(features_val, targets_val, args)
    data_test, targets_test = preprocess_data(features_test, targets_test, args)
    return adj, data_train, data_val, data_test, targets_train, targets_val, targets_test


def main(args):
    adj, data_train, data_val, data_test, targets_train, targets_val, targets_test = data_init(args)
    model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=50, silent=True, objective='reg:gamma')
    pres_test = np.zeros_like(targets_test)
    for tar_len, (target_train, target_test) in enumerate(zip(targets_train, targets_test)):
        for node, (t_train, t_test) in enumerate(zip(target_train, target_test)):
            model.fit(data_train, t_train)
            # 对测试集进行预测
            pre_test = model.predict(data_test)
            pres_test[tar_len, node, :] = pre_test

            print(pre_test)
            print("RMSE: ", rmse(pre_test.reshape(-1), t_test.reshape(-1)))
            print("MAE: ", mae(pre_test.reshape(-1), t_test.reshape(-1)))
            print("MARE: ", mare(pre_test.reshape(-1), t_test.reshape(-1)))
    print(sum(abs(pres_test.reshape(-1)-targets_test.reshape(-1))))
    print("RMSE: ", rmse(pres_test.reshape(-1), targets_test.reshape(-1)))
    print("MAE: ", mae(pres_test.reshape(-1), targets_test.reshape(-1)))
    print("MARE: ", mare(pres_test.reshape(-1), targets_test.reshape(-1)))



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

