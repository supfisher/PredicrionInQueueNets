import argparse
from utils import *
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
import numpy as np
from scipy import sparse
from sklearn.metrics import make_scorer
from sklearn.svm import SVR
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--cuda', action='store_true', default=False)
parser.add_argument('--file_head', type=str, default='svm',
                    help='which regression model to use.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--pre_len', type=int, default=10,
                    help='the length of input data sequence.')
parser.add_argument('--tar_len', type=int, default=3,
                    help='the length of output target sequence.')
parser.add_argument('--shuffle', action='store_true', default=True,
                    help='whether to shuffle the dataset.')

def data_init(args):
    adj, edge_index, features_train, features_val, features_test, \
        targets_train, targets_val, targets_test = load_dataset(path="./simul_data")

    if args.cuda:
        adj = adj.cuda()
        edge_index = edge_index.cuda()
    if len(targets_train.shape) == 1:
        targets_train = targets_train.unsqueeze(1)
        targets_val = targets_val.unsqueeze(1)
        targets_test = targets_test.unsqueeze(1)
    print("featues_train shape: ", features_train.shape)
    print("targets_train shape: ", targets_train.shape)
    print("adj shape: ", adj.shape)

    args.num_nodes = adj.shape[0]
    args.node_features = features_train.shape[-1]
    args.target_features = targets_train.shape[-1]

    return adj, edge_index, features_train, features_val, features_test, \
           targets_train, targets_val, targets_test


def data_loader(features, targets, args):
    """features:(len(times), node_feature)
    targets: (len(times), target_feature)
    retrun: (data_vev, target_vec)
    data_vev:(features.shape[0]-pre_len-tar_len, pre_len*node_feature)
    target_vec:(features.shape[0]-pre_len-tar_len, tar_len*target_feature)
    for simplicity we use node_features[0] as the target
    """
    data_vec = np.zeros([features.shape[0] - args.pre_len - args.tar_len, args.pre_len, features.shape[1]])
    target_vec = np.zeros([data_vec.shape[0], args.tar_len])
    if args.shuffle:
        orders = np.random.permutation(data_vec.shape[0])
    else:
        orders = range(data_vec.shape[0])
    for j, id in enumerate(orders):
        data_vec[j, :, 0:-1] = features[id:id+args.pre_len, 0:-1]
        data_vec[j, :, -1] = targets[id:id + args.pre_len, 0]

        target_vec[j, :] = targets[id+args.pre_len:id+args.pre_len+args.tar_len, 0]
    return data_vec.reshape(data_vec.shape[0], -1), target_vec


def data_init_numpy(args):
    adj, edge_index, features_train, features_val, features_test, \
    targets_train, targets_val, targets_test = data_init(args)

    return adj, features_train.numpy(), features_val.numpy(), features_test.numpy(), \
           targets_train.numpy(), targets_val.numpy(), targets_test.numpy()


def rmse(y_pre, y_true):
    y_pre = torch.from_numpy(y_pre)
    y_true = torch.from_numpy(y_true)
    inputs = y_pre - y_true
    return (torch.norm(inputs, p=2)/math.sqrt(inputs.shape[0])).item()


def mae(y_pre, y_true):
    y_pre = torch.from_numpy(y_pre)
    y_true = torch.from_numpy(y_true)
    inputs = y_pre - y_true
    return (torch.norm(inputs, p=1)/inputs.shape[0]).item()


def mad(y_pre, y_true):
    y_pre = torch.from_numpy(y_pre)
    y_true = torch.from_numpy(y_true)
    inputs = y_pre - y_true
    inputs = torch.abs(inputs - torch.median(inputs))
    return torch.median(inputs).item()


def R2(y_pre, y_true):
    y_pre = torch.from_numpy(y_pre)
    y_true = torch.from_numpy(y_true)
    mu = torch.mean(y_true)
    tmp = torch.pow(y_pre-y_true, 2).sum()/torch.pow(y_true-mu, 2).sum()
    return 1-tmp.item()



def main(args):
    adj, data_train, data_val, data_test, targets_train, targets_val, targets_test = data_init_numpy(args)

    min_target = min(targets_train)
    features_train, targets_train = data_loader(data_train, targets_train, args)
    features_test, targets_test = data_loader(data_test, targets_test, args)
    targets_train = targets_train - min_target
    targets_test = targets_test - min_target

    prediction_test = np.zeros_like(targets_test)

    for i in range(targets_train.shape[1]):
        # model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.01)
        model = xgb.XGBRegressor(max_depth=10, learning_rate=0.001,
                                 n_estimators=200, silent=True, objective="reg:gamma")
        model.fit(features_train, targets_train[:, i])
        model.score(features_test, targets_test[:, i])
        prediction_test[:, i] = model.predict(features_test)
        with open('XGBRegressor'+str(i)+'.pkl', 'wb') as f:
            pickle.dump(model, f)

    print("RMSE: ", rmse(prediction_test.reshape(-1), targets_test.reshape(-1)))
    print("MAE: ", mae(prediction_test.reshape(-1), targets_test.reshape(-1)))
    print("MAD: ", mad(prediction_test.reshape(-1), targets_test.reshape(-1)))
    print("R2: ", R2(prediction_test.reshape(-1), targets_test.reshape(-1)))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

