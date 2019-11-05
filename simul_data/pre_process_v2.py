"""we consider the inter_arrv_time with orders and inter_service_time with orders
why don't we put the inter_arrv_times into one single vector? Why do we need the orders?
For instance, we have several inter_arrv_times sorted by the arrv time. If we simplely put them into
one single vector, these features have temporal dependence, current ML tools is not good at solving features with
 temporal dependence. Therefore, we use the orders to remove the temporal dependence."""
import pandas as pd
import numpy as np
import math
# from threading import Thread as Multi
from multiprocessing import Process as Multi
import multiprocessing as mp
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


class Slide_data_thread(Multi):
    def __init__(self, id, q):
        Multi.__init__(self)
        self.id = id
        self.q = q
        kwargs = q.get()
        self.index_range = kwargs['index_range']
        self.datas = kwargs['data']
        self.arrv_times = kwargs['arrv_time']
        self.arv_arrays = kwargs['arv_arrays']
        self.sev_index_maps = kwargs['sev_index_maps']
        self.sev_indexs = kwargs['sev_indexs']
        self.sev_arrays = kwargs['sev_arrays']
        self.order = kwargs['order']
        self.num_nodes = kwargs['num_nodes']
        self.processed = kwargs['processed']

    def run(self):
        print("This is thread: ", self.id)
        features, features_dict, targets, targets_dic = self.slide_data_process(self.index_range, self.arrv_times, self.datas,
                                                                                self.arv_arrays, self.sev_index_maps,
                                                                                self.sev_indexs, self.sev_arrays,
                                                                                self.num_nodes, self.order, self.processed)
        self.q.put([features, features_dict, targets, targets_dic])


    def slide_data_process(self, index_range, arrv_times, datas, arv_arrays, sev_index_maps, sev_indexs, sev_arrays, num_nodes, order, processed):
        features = np.zeros((len(index_range), num_nodes,
                             order * 2 + 2))  # features are sorted by how many arrived but not served
        features_dict = {}
        targets = np.zeros(
            (len(index_range), 4))  ## we record the (q_id, arrv_time, service_time, response time)
        targets_dic = {}

        for j, (i, current_time, data) in enumerate(zip(index_range, arrv_times, datas)):
            features_tmp = np.zeros([num_nodes, order*2+2])

            for q in range(num_nodes):
                current_q = q
                index_arrv = binary_search(arv_arrays[current_q], current_time)
                index_maps = [sev_index_maps[current_q][id] for id in range(index_arrv+1)]
                sorted_index = sev_indexs[current_q][index_maps]
                sorted_prev_serv_array = sev_arrays[current_q][sorted_index] # store the sorted service time whose arrv time is less than current_time
                index_serv = binary_search(sorted_prev_serv_array, current_time)

                features_tmp[current_q, -2] = index_arrv - index_serv #how many arrival but not serve
                features_tmp[current_q, -1] = data[Index.q_id.value] # which queue does this log happen on?

                sorted_prev_arv_array = arv_arrays[current_q][index_arrv-order:index_arrv+1]
                sorted_prev_serv_array = sorted_prev_serv_array[-order-1:]
                if processed:
                    features_tmp[current_q, 0:0 + order] = order_inter_time(sorted_prev_arv_array)
                    features_tmp[current_q, 0 + order:0+order*2] = order_inter_time(sorted_prev_serv_array)
                else:
                    features_tmp[current_q, 0:0 + order] = _1order_inter_time(sorted_prev_arv_array)
                    features_tmp[current_q, 0 + order:0 + order * 2] = _1order_inter_time(sorted_prev_serv_array)

            features[j] = features_tmp
            features_dict[current_time] = features_tmp

            targets[j] = data[[Index.q_id.value, Index.arrival.value, Index.service.value, Index.departure.value]]
            targets_dic[current_time] = targets[j]
        return features, features_dict, targets, targets_dic


def slide_data(data, order=3, processed=True):
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
    features_dict = {}
    targets = np.zeros((len(data) - start_index, 4)) ## we record the (q_id, arrv_time, service_time, response time)
    targets_dic = {}

    num_threads = 30
    index_ranges = np.arange(0, len(arv_array)-start_index, int((len(arv_array)-start_index)/num_threads))
    threads = []
    queues = [mp.Manager().Queue() for _ in range(num_threads-1)]

    for i, q in enumerate(queues):
        index_range = np.arange(index_ranges[i], index_ranges[i+1])+start_index
        kwargs = {
            'index_range': index_range,
            'arrv_time': arv_array[index_range],
            'data': data[index_range],
            'arv_arrays': arv_arrays,
            'sev_index_maps': sev_index_maps,
            'sev_indexs': sev_indexs,
            'sev_arrays': sev_arrays,
            'num_nodes': num_nodes,
            'order': order,
            'processed': processed
        }
        q.put(kwargs)
        t = Slide_data_thread(i, q)
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for i, t in enumerate(threads):
        t_features, t_features_dict, t_targets, t_targets_dic = t.q.get()
        features_dict = {**features_dict, **t_features_dict}
        targets[index_ranges[i]:index_ranges[i+1]] = t_targets
        targets_dic = {**targets_dic, **t_targets_dic}

    return features_dict, targets, targets_dic, q_ids


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

    file_head = 'weight_'+str(weight)+'_height_'+str(height)
    # file_head = 'pagerank'

    data = load_data(file_head+'_queue.csv')
    features_dict, targets, targets_dic, q_ids = slide_data(data, order=2)
    print("length of valid queues: ", len(q_ids))
    adj = load_data(file_head + '_adj.csv')
    adj += adj.T
    print('shape of original adj: ', adj.shape)
    adj = adj[q_ids][:, q_ids]
    print('shape of processed adj: ', adj.shape)

    data2pickle(features_dict, './features.pkl')
    data2csv(adj, './adj.csv')
    data2pickle(targets_dic, './targets.pkl')
    data2csv(targets, './targets.csv')

