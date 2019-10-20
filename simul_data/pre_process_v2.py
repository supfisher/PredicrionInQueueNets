"""we consider the inter_arrv_time with orders and inter_service_time with orders
why don't we put the inter_arrv_times into one single vector? Why do we need the orders?
For instance, we have several inter_arrv_times sorted by the arrv time. If we simplely put them into
one single vector, these features have temporal dependence, current ML tools is not good at solving features with
 temporal dependence. Therefore, we use the orders to remove the temporal dependence."""
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
    print(data.values[0:5])
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


def slide_data(data, num_nodes, order=3, processed=True):
    """a combination of slide_window and slide_time
     this function reads each log from the sorted arrival time log dataset: data,
     and generate features from the data[current-slide: current],
     features:[# arrival events in the period, # departure events in the period, # arrival events but departure in the period,
                inter_arrv_time, inter_dept_time]
     the features shape is (len(data)-step, num_node, 3+2*orders)"""


    arv_array = np.array(data[:, Index.arrival.value])
    q_ids = np.array(data[:, Index.q_id.value])

    arv_list = [[] for _ in range(num_nodes)]
    sev_list = [[] for _ in range(num_nodes)]
    for i, d in enumerate(data):
        arv_list[int(d[Index.q_id.value])].append(d[Index.arrival.value])
        sev_list[int(d[Index.q_id.value])].append(d[Index.service.value])

    arv_arrays = [np.array(arv) for arv in arv_list]
    sev_arrays = [np.array(sev) for sev in sev_list]

    sev_indexs = [np.argsort(sev_arrays[i]) for i in range(num_nodes)]
    sev_index_maps = {i: {sev_indexs[i][j]: j for j in range(len(sev_indexs[i]))} for i in range(num_nodes)}

    orders_list = [binary_search(arv_array, arv_arrays[i][order]) for i in range(num_nodes)]
    start_index = max(orders_list)
    features = np.zeros((len(data) - start_index, num_nodes, 1+order*2)) #features are sorted by how many arrived but not served
    features_dict = {}
    targets = np.zeros((len(data) - start_index, 4)) ## we record the (q_id, arrv_time, service_time, response time)
    targets_dic = {}

    for i in range(start_index, len(arv_array)):
        features_tmp = np.zeros([num_nodes, 1+order*2])
        current_time = arv_array[i]

        for q in range(num_nodes):
            current_q = q
            index_arrv = binary_search(arv_arrays[current_q], current_time)
            index_maps = [sev_index_maps[current_q][id] for id in range(index_arrv+1)]
            sorted_index = sev_indexs[current_q][index_maps]
            sorted_prev_serv_array = sev_arrays[current_q][sorted_index] # store the sorted service time whose arrv time is less than current_time
            index_serv = binary_search(sorted_prev_serv_array, current_time)

            features_tmp[current_q, 0] = index_arrv - index_serv

            sorted_prev_arv_array = arv_arrays[current_q][index_arrv-order:index_arrv+1]
            sorted_prev_serv_array = sorted_prev_serv_array[-order-1:]
            if processed:
                features_tmp[current_q, 1:1 + order] = order_inter_time(sorted_prev_arv_array)
                features_tmp[current_q, 1 + order:1+order*2] = order_inter_time(sorted_prev_serv_array)
            else:
                features_tmp[current_q, 1:1 + order] = _1order_inter_time(sorted_prev_arv_array)
                features_tmp[current_q, 1 + order:1 + order * 2] = _1order_inter_time(sorted_prev_serv_array)

        features[i - start_index] = features_tmp
        features_dict[current_time] = features_tmp

        targets[i - start_index] = data[i][[Index.q_id.value, Index.arrival.value, Index.service.value, Index.departure.value]]
        targets_dic[current_time] = targets[i - start_index]
    return features, features_dict, targets, targets_dic


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

    if weight != 1:
        num_nodes = int((math.pow(weight, height+1)-1)/(weight-1))
    else:
        num_nodes = height
    data = load_data('weight_'+str(weight)+'_height_'+str(height)+'_agent_queue.csv')
    data = data[int(len(data)/5):]
    features, features_dict, targets, targets_dic = slide_data(data, num_nodes)
    print(features.shape)
    print(features[-10:-1])
    adj = load_data('weight_' + str(weight) + '_height_' + str(height) + '_adj.csv')
    adj += adj.T
    data2pickle(features_dict, './features.pkl')
    data2csv(adj, './adj.csv')
    data2pickle(targets_dic, './targets.pkl')
    data2csv(targets, './targets.csv')

