"""we consider the inter_arrv_time with orders and inter_service_time with orders
why don't we put the inter_arrv_times into one single vector? Why do we need the orders?
For instance, we have several inter_arrv_times sorted by the arrv time. If we simplely put them into
one single vector, these features have temporal dependence, current ML tools is not good at solving features with
 temporal dependence. Therefore, we use the orders to remove the temporal dependence."""
"""In this version, we only consider about one queue node for one time instance"""
import pandas as pd
import numpy as np
import math
import os
from threading import Thread as Multi
# from multiprocessing import Process as Multi
import multiprocessing as mp
from enum import Enum
import torch.distributed as dist
import torch
import random

class Index(Enum):
    arrival = 0
    service = 1
    departure = 2
    num_queued = 3
    num_total = 4
    q_id = 5

def binary_search(arry, value):
    """return: the the index i of a sorted arry with arry[i]<=value<arry[i+1] """
    low = 0
    high = len(arry)
    while high - low > 1:
        mid = math.ceil((high+low)/2)
        if arry[mid] > value:
            high = mid
        elif arry[mid] <= value:
            low = mid
    return low


def load_data(path):
    data = pd.read_csv(path)
    # print(data.values[0:5])
    return data.values


def _1order_inter_time(in_time_array):
    tmp=[]
    for i in range(len(in_time_array)-1):
        tmp.append(in_time_array[i+1]-in_time_array[i])
    return sum(tmp)/len(tmp), tmp


def order_inter_time(time_array):
    """this function returns the difference ordered array given a time array"""

    diff_ordered_values = []
    tmp = time_array
    for i in range(len(time_array)-1):
        v, tmp = _1order_inter_time(tmp)
        diff_ordered_values.append(v)
    return diff_ordered_values


def slide_data_process(index_range, arrv_times, datas, arv_arrays, sev_index_maps, sev_indexs, sev_arrays, q_ids2index, order, processed):
    features = np.zeros((len(index_range), order * 2 + 3))  # features are sorted by how many arrived but not served
    features_dict = {}
    targets = np.zeros(
        (len(index_range), 4))  ## we record the (q_id, arrv_time, service_time, response time)
    targets_dic = {}

    for j, (i, current_time, data) in enumerate(zip(index_range, arrv_times, datas)):
        features_tmp = np.zeros([order*2+3])

        current_q = q_ids2index[data[Index.q_id.value]]

        index_arrv = binary_search(arv_arrays[current_q], current_time)
        index_maps = [sev_index_maps[current_q][id] for id in range(index_arrv+1)]
        sorted_index = sev_indexs[current_q][index_maps]
        sorted_prev_serv_array = sev_arrays[current_q][sorted_index] # store the sorted service time whose arrv time is less than current_time
        index_serv = binary_search(sorted_prev_serv_array, current_time)

        features_tmp[-3] = index_arrv - index_serv #how many arrival but not serve
        features_tmp[-2] = data[Index.q_id.value] # which queue does this log happen on?
        features_tmp[-1] = current_q

        sorted_prev_arv_array = arv_arrays[current_q][index_arrv-order:index_arrv+1]
        sorted_prev_serv_array = sorted_prev_serv_array[index_serv-order:index_serv+1]  #TODO: modified the sorted_prev_serv_array
        if processed:
            features_tmp[0:0 + order] = order_inter_time(sorted_prev_arv_array)
            features_tmp[0 + order:0+order*2] = order_inter_time(sorted_prev_serv_array)
        else:
            features_tmp[0:0 + order] = _1order_inter_time(sorted_prev_arv_array)
            features_tmp[0 + order:0 + order * 2] = _1order_inter_time(sorted_prev_serv_array)

        features[j] = features_tmp
        features_dict[current_time] = features_tmp

        targets[j] = data[[Index.q_id.value, Index.arrival.value, Index.service.value, Index.departure.value]]
        targets_dic[current_time] = targets[j]
    return features, features_dict, targets, targets_dic


def slide_data(data, rank, world_size, order=3, processed=True):
    """a combination of slide_window and slide_time
     this function reads each log from the sorted arrival time log dataset: data,
     and generate features from the data[current-slide: current],
     features:[# arrival events in the period, # departure events in the period, # arrival events but departure in the period,
                inter_arrv_time, inter_dept_time]
     the features shape is (len(data)-step, num_node, 3+2*orders)"""


    arv_array = np.array(data[:, Index.arrival.value])
    q_ids = [int(q_id) for q_id in set(np.array(data[:, Index.q_id.value]))]

    num_nodes = len(q_ids)
    q_ids2index = {}
    for i, q in enumerate(q_ids):
        q_ids2index[q] = i

    arv_list = [[] for _ in range(num_nodes)]
    sev_list = [[] for _ in range(num_nodes)]
    for i, d in enumerate(data):
        arv_list[q_ids2index[int(d[Index.q_id.value])]].append(d[Index.arrival.value])
        sev_list[q_ids2index[int(d[Index.q_id.value])]].append(d[Index.service.value])

    arv_arrays = [np.array(arv) for arv in arv_list]
    sev_arrays = [np.array(sev) for sev in sev_list]

    sev_indexs = [np.argsort(sev_arrays[i]) for i in range(num_nodes)]
    sev_index_maps = {i: {sev_indexs[i][j]: j for j in range(len(sev_indexs[i]))} for i in range(num_nodes)}

    orders_list = [binary_search(arv_array, arv_arrays[i][order]) for i in range(num_nodes)]
    start_index = max(orders_list)

    index_ranges = np.arange(0, len(arv_array)-start_index+1, int((len(arv_array)-start_index)/world_size))

    index_range = np.arange(index_ranges[rank], index_ranges[rank+1])+start_index


    _, features_dict, targets, targets_dic = slide_data_process(index_range, arv_array[index_range], data[index_range],
                                                                arv_arrays, sev_index_maps, sev_indexs, sev_arrays, q_ids2index, order, processed)

    times = arv_array[np.arange(index_ranges[0], index_ranges[world_size-1])+start_index]
    features_train, features_val, features_test, targets_train, targets_val, \
        targets_test = train_test_split(features_dict, targets_dic, times, ratio=(0.5, 0.3,))
    print("rank: ", rank, "shape of train features: ", features_train.shape, "shape of train targets: ", targets_train.shape)
    return features_train, features_val, features_test, targets_train, targets_val, \
        targets_test, q_ids


def normalization(data, mu=None, std=None, method='feat'):
    """the first dimision of data is len(times)"""

    if method == 'feat':
        if mu is None and std is None:
            mu = torch.mean(data[:, 0:-1], dim=0)
            std = torch.std(data[:, 0:-1], dim=0)
            std[torch.where(std == 0)] = 1
        data[:, 0:-1] = (data[:, 0:-1] - mu) / std
    else:
        if mu is None and std is None:
            mu = torch.mean(data, dim=0)
            std = torch.std(data, dim=0)
        data = (data - mu) / std

        # min = torch.min(data)
        # max = torch.max(data)
        # data = (data-min) / (max-min)

    return data, mu, std
# def normalization(data, min=None, max=None, method='feat'):
#     """the first dimision of data is len(times)"""
#
#     if method == 'feat':
#         if min is None and max is None:
#             min = torch.min(data[:, 0:-1], dim=0)[0]
#             max = torch.max(data[:, 0:-1], dim=0)[0]
#             max[torch.where(max == 0)] = 1
#         data[:, 0:-1] = (data[:, 0:-1] - min) / (max - min)
#     else:
#         if min is None and max is None:
#             min = torch.min(data, dim=0)[0]
#             max = torch.max(data, dim=0)[0]
#         data = (data - min) / (max - min)
#
#     return data, min, max


def train_test_split(features, targets, times, ratio=(0.5, 0.3, ), sample_rate=None, constant_step=True):
    if sample_rate is not None:
        if constant_step:
            index = np.arange(0, len(times), int(1 / sample_rate))
            times = times[index]
        else:
            times = random.sample(times, int(len(times) * sample_rate))
    Id = {'q_id': 0, 'arrival': 1, 'service': 2, 'departure': 3}


    train_time_bound = times[int(len(times) * ratio[0])-1]
    val_time_bound = times[int(len(times) * (ratio[0] + ratio[1]))-1]

    features_time = np.array(list(features.keys()))
    train_index = features_time <= train_time_bound
    val_index = [q and p for q, p in zip(features_time > train_time_bound, features_time <= val_time_bound)]
    test_index = features_time > val_time_bound

    features_train = torch.tensor([features[i] for i in features_time[train_index]])
    features_val = torch.tensor([features[i] for i in features_time[val_index]])
    features_test = torch.tensor([features[i] for i in features_time[test_index]])
    # print("features train shape: ", features_train.shape)
    targets_train = torch.tensor([targets[i][Id['service']] - targets[i][Id['arrival']] for i in features_time[train_index]])
    targets_val = torch.tensor([targets[i][Id['service']] - targets[i][Id['arrival']] for i in features_time[val_index]])
    targets_test = torch.tensor([targets[i][Id['service']] - targets[i][Id['arrival']] for i in features_time[test_index]])
    # targets_train = torch.tensor(
    #     [targets[i][Id['arrival']] for i in features_time[train_index]])
    # targets_val = torch.tensor(
    #     [targets[i][Id['arrival']] for i in features_time[val_index]])
    # targets_test = torch.tensor(
    #     [targets[i][Id['arrival']] for i in features_time[test_index]])

    return features_train, features_val, features_test, targets_train, targets_val, targets_test


def data2csv(data_list, path):
    import pandas as pd
    data_pd = pd.DataFrame(data=data_list)
    data_name = path
    data_pd.to_csv(data_name, index=False)


def data2pickle(data_dic, path, mode='wb'):
    import pickle
    with open(path, mode) as f:
        pickle.dump(data_dic, f)


def file_combine(path, word_size, mu=None, std=None, method='feat'):
    def gen(word_size):
        for i in range(word_size):
            if os.path.exists(path + 'ws_' + str(word_size) + 'rank_' + str(i)):
                with open(path + 'ws_' + str(word_size) + 'rank_' + str(i), 'rb') as f:
                    yield torch.load(f)
    out = torch.cat([data for data in gen(word_size)], 0)
    print("concated shape: ", path, out.shape)

    out_mu = None
    out_std = None
    out, out_mu, out_std = normalization(out, mu, std, method=method)
    print("out_mu: ", out_mu)
    print("out_std: ", out_std)
    torch.save(out, path)
    return out, out_mu, out_std


def extend_features(features, targets, q_num, current_adj_feat=None):
    """
    This function is used to extend the log into and adj features
    :param features: (len, feature_len)
    :param targets: (len, target_len)
    :param q_num: (numof queues)
    :return: features_extend: (len adj_shape, feature)
    """
    data_len, feat_len = features.shape
    features_extend = torch.zeros(data_len, q_num, feat_len+1)

    if current_adj_feat is None:
        current_adj_feat = torch.zeros(q_num, feat_len+1)

    for i in range(data_len):
        q_id = features[i, -1].long()
        current_adj_feat[q_id, 0:-2] = features[i, 0:-1]
        current_adj_feat[q_id, -2] = targets[i]
        current_adj_feat[:, -1] = q_id
        features_extend[i] = current_adj_feat

    return features_extend, current_adj_feat


def features_resort(features, targets, q_num):
    '''
    This function is used to re-sort the features and targets according to the queue id and for each
    queue, it is sorted by arrival time.
    :param features:
    :param targets:
    :param q_num:
    :return:
    '''
    features_q = [[] for _ in range(q_num)]
    targets_q = [[] for _ in range(q_num)]
    for i, (feat, target) in enumerate(zip(features, targets)):
        q = int(feat[0, -1])
        features_q[q].append(feat)
        targets_q[q].append(target)
    features = []
    targets = []
    for features_1q, targets_1q in zip(features_q, targets_q):
        features.append(torch.stack(features_1q))
        targets.append(torch.stack(targets_1q))

    return features, targets


def rebuild_features(features, targets, q_num, current_adj_feat=None, path='train.pkl'):
    features_extend, current_adj_feat = extend_features(features,
                                                        targets, q_num, current_adj_feat=current_adj_feat)

    features, targets = features_resort(features_extend, targets, q_num)
    torch.save(features, './features_adj_'+path)
    torch.save(targets, './targets_adj_' + path)

    return current_adj_feat



def clear(rank, word_size):
    os.system("rm " + './features_train.pkl' + 'ws_' + str(word_size) + 'rank_' + str(rank))
    os.system("rm " + './features_val.pkl' + 'ws_' + str(word_size) + 'rank_' + str(rank))
    os.system("rm " + './features_test.pkl' + 'ws_' + str(word_size) + 'rank_' + str(rank))
    os.system("rm " + './targets_train.pkl' + 'ws_' + str(word_size) + 'rank_' + str(rank))
    os.system("rm " + './targets_val.pkl' + 'ws_' + str(word_size) + 'rank_' + str(rank))
    os.system("rm " + './targets_test.pkl' + 'ws_' + str(word_size) + 'rank_' + str(rank))
    os.system("rm " + './meta_data.pkl' + 'ws_' + str(word_size) + 'rank_' + str(rank))


def main_worker(file_head):
    data = load_data(file_head + '_queue.csv')
    rank = dist.get_rank()
    word_size = dist.get_world_size()
    features_train, features_val, features_test, targets_train, \
    targets_val, targets_test, q_ids = slide_data(data, rank, word_size, order=3)

    if features_train.shape[0] > 0:
        torch.save(features_train, './features_train.pkl' + 'ws_' + str(word_size) + 'rank_' + str(rank))
        torch.save(targets_train, './targets_train.pkl' + 'ws_' + str(word_size) + 'rank_' + str(rank))
    if features_val.shape[0] > 0:
        torch.save(features_val, './features_val.pkl' + 'ws_' + str(word_size) + 'rank_' + str(rank))
        torch.save(targets_val, './targets_val.pkl' + 'ws_' + str(word_size) + 'rank_' + str(rank))
    if features_test.shape[0] > 0:
        torch.save(features_test, './features_test.pkl' + 'ws_' + str(word_size) + 'rank_' + str(rank))
        torch.save(targets_test, './targets_test.pkl' + 'ws_' + str(word_size) + 'rank_' + str(rank))

    sync = torch.zeros(1)
    dist.all_reduce(sync, async_op=False)

    print(dist.get_rank(), ": file split finished, begin to combine")

    if dist.get_rank() == 0:
        features_train, mu, std = file_combine('./features_train.pkl', word_size, method='feat')
        features_val, _, _ = file_combine('./features_val.pkl', word_size, mu=mu, std=std, method='feat')
        features_test, _, _ = file_combine('./features_test.pkl', word_size, mu=mu, std=std, method='feat')

        targets_train, mu, std = file_combine('./targets_train.pkl', word_size, method='targ')
        targets_test, _, _ = file_combine('./targets_test.pkl', word_size, mu=mu, std=std, method='targ')
        targets_val, _, _ = file_combine('./targets_val.pkl', word_size, mu=mu, std=std, method='targ')

        print("length of valid queues: ", len(q_ids))
        adj = load_data(file_head + '_adj.csv')
        adj += adj.T

        print('shape of original adj: ', adj.shape)
        adj = adj[q_ids][:, q_ids]
        print('shape of processed adj: ', adj.shape)
        data2csv(adj, './adj.csv')

        q_num = adj.shape[0]
        current_adj_feat = rebuild_features(features_train, targets_train,
                                            q_num, current_adj_feat=None, path='train.pkl')
        current_adj_feat = rebuild_features(features_val, targets_val,
                                            q_num, current_adj_feat=current_adj_feat, path='val.pkl')
        current_adj_feat = rebuild_features(features_test, targets_test,
                                            q_num, current_adj_feat=current_adj_feat, path='test.pkl')


    sync = torch.zeros(1)
    dist.all_reduce(sync, async_op=False)
    clear(rank, word_size)



if __name__=='__main__':
    dist.init_process_group(backend='mpi', init_method="./")
    # weight = 1
    # height = 5
    # file_head = 'weight_'+str(weight)+'_height_'+str(height)
    file_head = 'pagerank'
    main_worker(file_head)


