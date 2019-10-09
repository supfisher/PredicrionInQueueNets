import numpy as np
from sklearn.linear_model import LinearRegression as LR
import xgboost as xgb
from xgboost import plot_importance
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import math

def rmse(y_pre, y_true):
    inputs = y_pre - y_true
    return np.linalg.norm(inputs, ord=2)/math.sqrt(len(inputs))

def mae(y_pre, y_true):
    inputs = y_pre - y_true
    return np.linalg.norm(inputs, ord=1)/len(inputs)

def mare(y_pre, y_true):
    inputs = [(i-j)/j for i, j in zip(y_pre, y_true)]
    return np.linalg.norm(inputs, ord=1)/len(inputs)


def preprocess_data(data, rate=0.8, seq_len=5, pre_len=3):
    data1 = list(data.values)
    time_len = len(data1)
    train_size = int(time_len * rate)
    train_data = data1[0:train_size]
    test_data = data1[train_size:time_len]

    trainX, trainY, testX, testY = [], [], [], []
    for i in range(len(train_data) - seq_len - pre_len):
        a = train_data[i: i + seq_len + pre_len]
        trainX.append(a[0: seq_len])
        trainY.append(a[seq_len: seq_len + pre_len])
    for i in range(len(test_data) - seq_len - pre_len):
        b = test_data[i: i + seq_len + pre_len]
        testX.append(b[0: seq_len])
        testY.append(b[seq_len: seq_len + pre_len])
    return trainX, trainY, testX, testY


def main(path):
    data = pd.read_csv(path)
    seq_len = 5
    pre_len = 1
    a_X, a_Y, t_X, t_Y = preprocess_data(data, rate=0.8, seq_len=5, pre_len=1)
    a_X = np.array(a_X)
    a_X = np.reshape(a_X, [-1, seq_len])
    a_Y = np.array(a_Y)
    a_Y = np.reshape(a_Y, [-1, pre_len])
    a_Y = np.mean(a_Y, axis=1)
    t_X = np.array(t_X)
    t_X = np.reshape(t_X, [-1, seq_len])
    t_Y = np.array(t_Y)
    t_Y = np.reshape(t_Y, [-1, pre_len])
    model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, silent=True, objective='reg:gamma')
    svr_model = SVR(kernel='linear')
    svr = svr_model.fit(a_X, a_Y)
    print("svr: ", svr.score(a_X, a_Y))
    reg = model.fit(a_X, a_Y)
    print("reg: ", reg.score(a_X, a_Y))
    # 对测试集进行预测
    y_pre = model.predict(t_X)
    pre = svr_model.predict(t_X)
    print("RMSE: ", rmse(y_pre, t_Y))
    print("MAE: ", mae(y_pre, t_Y))
    print("MARE: ", mare(y_pre, t_Y))
    plt.plot(y_pre, 'r')
    plt.plot(t_Y, 'b')
    plt.show()
    plt.plot(model.predict(a_X), 'r')
    plt.plot(a_Y, 'b')
    plt.show()
    # 显示重要特征
    plot_importance(model)
    plt.show()


if __name__ == "__main__":
    path = 'weight_3_height_2_agent_queue.csv'
    main(path)

