import pandas as pd
import numpy as np
import math
from enum import Enum

class Index(Enum):
    arrival = 0
    service = 1
    departure = 2
    num_queued = 3
    num_total = 4
    q_id = 5

def binary_search(arry, value):
    """return: the the index i of a sorted arry which value>arry[i] and value<=arry[i+1] """
    low = 0
    high = len(arry) - 1
    while high - low > 1:
        mid = (high+low)//2
        if arry[mid] > value:
            high = mid
        elif arry[mid] <= value:
            low = mid
    return low+1


def load_data(path):
    data = pd.read_csv(path)
    print(data.values[0:5])
    return data.values


def slide_time(data, num_nodes, time_step=None, sample_rate=0.01):
    """from the original log data to generate a dic, where key is the time, value is the
     list including the data from current index-sliding window to current index """
    arv_array = data[:, Index.arrival.value]
    depar_array = np.array(data[:, Index.departure.value])
    min_arv_time = min(arv_array)
    max_arv_time = max(arv_array)
    if time_step is None:
        time_step = ((max_arv_time-min_arv_time)*sample_rate)
    start_index = binary_search(arv_array, time_step)

    features = np.zeros((len(arv_array)-start_index, num_nodes, 3))
    features_dict = {}

    for i in range(start_index, len(arv_array)):
        "to save the computation memory, once we get a piece of event logs, we tranfer it to features"
        arv_data = {}
        depart_data = {}
        arv_depart_data = {}

        current_time = arv_array[i]
        s_time = current_time - time_step

        s_index = binary_search(arv_array, s_time)
        arv_indexes = list(range(s_index, i))
        arv_data[current_time] = data[arv_indexes, :]

        current_depart_indexes = np.where(depar_array <= current_time)[0].tolist()

        s_depart_indexes = np.where(depar_array < s_time)[0].tolist()
        depart_indexes = list(set(current_depart_indexes).difference(set(s_depart_indexes)))
        depart_data[current_time] = data[depart_indexes, :]

        inter_indexes = list(set(arv_indexes).intersection(set(depart_indexes)))
        arv_depart_data[current_time] = data[inter_indexes, :]

        features_tmp, features_dict_tmp = logs_to_features(arv_data, depart_data, arv_depart_data, num_nodes)

        features[i-start_index,:,:] += features_tmp[0,:,:]
        features_dict[current_time] = features_dict_tmp[current_time]
    return features, features_dict


def slide_window(data, num_nodes, wind_step=20):
    """from the original log data to generate a dic, where key is the time, value is the
         list including the data from current index-sliding window to current index """
    arv_array = data[:, Index.arrival.value]
    depar_array = np.array(data[:, Index.departure.value])
    start_index = wind_step

    features = np.zeros((len(arv_array) - start_index, num_nodes, 3))
    features_dict = {}

    for i in range(start_index, len(arv_array)):
        "to save the computation memory, once we get a piece of event logs, we tranfer it to features"
        arv_data = {}
        depart_data = {}
        arv_depart_data = {}

        current_time = arv_array[i]
        s_index = i - start_index
        s_time = arv_array[s_index]

        arv_indexes = list(range(s_index, i))
        arv_data[current_time] = data[arv_indexes, :]

        current_depart_indexes = np.where(depar_array <= current_time)[0].tolist()

        s_depart_indexes = np.where(depar_array < s_time)[0].tolist()
        depart_indexes = list(set(current_depart_indexes).difference(set(s_depart_indexes)))
        depart_data[current_time] = data[depart_indexes, :]

        inter_indexes = list(set(arv_indexes).intersection(set(depart_indexes)))
        arv_depart_data[current_time] = data[inter_indexes, :]

        features_tmp, features_dict_tmp = logs_to_features(arv_data, depart_data, arv_depart_data, num_nodes)

        features[i - start_index, :, :] += features_tmp[0, :, :]
        features_dict[current_time] = features_dict_tmp[current_time]
    return features, features_dict


def slide_data(data, num_nodes, wind_step=None, time_step=None, sample_rate=0.01):
    """a combination of slide_window and slide_time
     this function reads each log from the sorted arrival time log dataset: data,
     and generate features from the data[current-slide: current],
     features:[# arrival events in the period, # departure events in the period, # arrival events but departure in the period]
     the features shape is (len(data)-step, num_node, 3)"""
    arv_array = np.array(data[:, Index.arrival.value])
    depar_array = np.array(data[:, Index.departure.value])

    min_arv_time = min(arv_array)
    max_arv_time = max(arv_array)
    if wind_step is None:
        if time_step is None:
            time_step = ((max_arv_time - min_arv_time) * sample_rate)
        start_index = binary_search(arv_array, time_step)
    elif wind_step is not None:
        start_index = wind_step

    features = np.zeros((len(arv_array) - start_index, num_nodes, 3))
    features_dict = {}
    targets = np.zeros((len(arv_array) - start_index, 3)) ## we record the (q_id, service_time, response time)
    targets_dic = {}
    for i in range(start_index, len(arv_array)):
        "to save the computation memory, once we get a piece of event logs, we tranfer it to features"
        arv_data = {}
        depart_data = {}
        arv_depart_data = {}
        current_time = arv_array[i]
        if wind_step is None:
            s_time = current_time - time_step
            s_index = binary_search(arv_array, s_time)
        else:
            s_index = i - start_index
            s_time = arv_array[s_index]

        arv_indexes = list(range(s_index, i+1))
        arv_data[current_time] = data[arv_indexes, :]

        current_depart_indexes = np.where(depar_array <= current_time)[0].tolist()

        s_depart_indexes = np.where(depar_array < s_time)[0].tolist()
        depart_indexes = list(set(current_depart_indexes).difference(set(s_depart_indexes)))
        depart_data[current_time] = data[depart_indexes, :]

        inter_indexes = list(set(arv_indexes).intersection(set(depart_indexes)))
        arv_depart_data[current_time] = data[inter_indexes, :]

        features_tmp, features_dict_tmp = logs_to_features(arv_data, depart_data, arv_depart_data, num_nodes)

        features[i - start_index, :, :] += features_tmp[0, :, :]
        features_dict[current_time] = features_dict_tmp[current_time]

        targets[i - start_index, 0] = data[i, Index.q_id.value]
        targets[i - start_index, 1] = data[i, Index.departure.value] - data[i, Index.service.value]
        targets[i - start_index, 2] = data[i, Index.departure.value] - data[i, Index.arrival.value]
        targets_dic[current_time] = targets[i - start_index]
    return features, features_dict, targets, targets_dic



def uniform_split_time(data, sample_rate):
    """split the event logs data to many small pieces"""
    arv_array = np.array(data[:, Index.arrival.value])
    depar_array = np.array(data[:, Index.departure.value])
    min_arv_time = min(arv_array)
    max_arv_time = max(arv_array)
    time_slices = list(np.arange(min_arv_time, max_arv_time, (max_arv_time-min_arv_time)*sample_rate))
    time_slices.append(max_arv_time)

    last_arrv_index = 0
    arv_data = {}
    last_depart_indexes = [0]
    depart_data = {}
    arv_depart_data = {}
    for i, t in enumerate(time_slices):
        if i == 0:
            continue
        arv_index = binary_search(arv_array[last_arrv_index:], t) + last_arrv_index
        arv_indexes = list(range(last_arrv_index,arv_index))
        arv_data[t] = data[arv_indexes, :]
        last_arrv_index = arv_index

        current_depart_indexes = np.where(depar_array <= t)[0].tolist()
        depart_indexes = list(set(current_depart_indexes).difference(set(last_depart_indexes)))
        depart_data[t] = data[depart_indexes,:]
        last_depart_indexes = current_depart_indexes

        inter_indexes = list(set(arv_indexes).intersection(set(depart_indexes)))
        arv_depart_data[t] = data[inter_indexes,:]

    return arv_data, depart_data, arv_depart_data


def logs_to_features(arv_data, depart_data, arv_depart_data, num_nodes):
    features = np.zeros((len(arv_data), num_nodes, 3))
    features_dict = {}
    times = sorted(arv_data.keys())
    for i, time in enumerate(times):
        for d in arv_data[time]:
            q_id = int(d[Index.q_id.value])
            features[i, q_id, 0] += 1
        for d in depart_data[time]:
            q_id = int(d[Index.q_id.value])
            features[i, q_id, 1] += 1
        for d in arv_depart_data[time]:
            q_id = int(d[Index.q_id.value])
            features[i, q_id, 2] += 1
        features_dict[time] = features[i, :, :]
    return features, features_dict


def data2csv(data_list, path):
    import pandas as pd
    data_pd = pd.DataFrame(data=data_list)
    data_name = path
    data_pd.to_csv(data_name, index=False)


def data2pickle(data_dic, path):
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(data_dic, f)


if __name__=='__main__':
    weight = 1
    height = 5
    window_size = 500
    sample_rate = 0.1
    num_nodes = height
    if weight != 1:
        num_nodes = int((math.pow(weight, height+1)-1)/(weight-1))
    data = load_data('weight_'+str(weight)+'_height_'+str(height)+'_agent_queue.csv')
    features, features_dict, targets, targets_dic = slide_data(data, num_nodes, wind_step=window_size, time_step=None, sample_rate=0.01)
    print(features.shape)
    print(features[-10:-1])
    adj = load_data('weight_' + str(weight) + '_height_' + str(height) + '_adj.csv')
    adj += adj.T
    data2pickle(features_dict, './features.pkl')
    data2csv(adj, './adj.csv')
    data2pickle(targets_dic, './targets.pkl')
    data2csv(targets, './targets.csv')

