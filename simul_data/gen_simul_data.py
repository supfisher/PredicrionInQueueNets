import queueing_tool as qt
import numpy as np
import copy
from collections import Counter
import argparse
parser = argparse.ArgumentParser(description='build a simulate queue network')
from enum import Enum

class Index(Enum):
    arrival = 0
    service = 1
    departure = 2
    num_queued = 3
    num_total = 4
    q_id = 5


def line_queue(length, entry):
    adja_list = {}
    edge_list = {}
    for i in range(length):
        adja_list[i] = [i+1]
        edge_list[i] = {i+1: i+1}
    (a, b) = entry
    edge_list[a][b] = 1
    return adja_list, edge_list

def binary_tree_queue(height, entry):
    adja_list = {}
    edge_list = {}
    total_num = pow(2, height)
    curr_height = 1
    for i in range(total_num):
        if i == 0:
            adja_list[0] = [1]
        else:
            adja_list[i] = [2*i, 2*i+1]
        if i >= pow(2, curr_height):
            curr_height += 1
        edge_list[i] = {j: curr_height+1 for j in adja_list[i]}
    (a, b) = entry
    edge_list[a][b] = 1
    return adja_list, edge_list

def mul_tree_queue(weight, height, entry):
    adja_list = {}
    edge_list = {}
    total_num = int((pow(weight, height) - 1)/(weight-1)) + 1
    curr_height = 1
    for i in range(total_num):
        if i == 0:
            adja_list[0] = [1]
        else:
            adja_list[i] = [j for j in range(weight*(i-1)+2, weight*i+2)]
        if i >= (pow(weight, curr_height)-1)/(weight-1)+1:
            curr_height += 1
        edge_list[i] = {j: curr_height+1 for j in adja_list[i]}
    (a, b) = entry
    edge_list[a][b] = 1
    return adja_list, edge_list


def gen_tree(weight, height, entry):
    if weight == 1:
        return line_queue(height, entry)
    elif weight == 2:
        return binary_tree_queue(height, entry)
    else:
        return mul_tree_queue(weight, height, entry)


def rate(t):
    return 5 + 35 * np.sin(np.pi * t / 2)**2


def arr_f(t):
    return qt.poisson_random_measure(t, rate, 4)

def ser_f(t):
    return t + np.random.exponential(0.3/args.weight)


def graph2queue(adja_list, edge_list, args):
    q_classes = {1: qt.QueueServer, 2: qt.QueueServer}
    for i in range(3, args.height+2):
        q_classes[i] = qt.QueueServer
    print("q_classes: ", q_classes)
    q_args = {
        1: {
            'arrival_f': arr_f,
            'service_f': lambda t: t+np.random.exponential(0.1),
            'AgentFactory': qt.GreedyAgent
        },
        2: {
            'num_servers': 3,
            'service_f': ser_f,
            'AgentFactory': qt.GreedyAgent
        }
    }
    for i in range(3, args.height+2):
        q_args[i] = {'num_servers': 3,
                        'service_f': lambda t: t + np.random.exponential(0.3*(args.weight**(i-3))),
                        'AgentFactory': qt.GreedyAgent}
    print("q_args: ", q_args)
    g = qt.adjacency2graph(adjacency=adja_list, edge_type=edge_list)
    qn = qt.QueueNetwork(g=g, q_classes=q_classes, q_args=q_args, seed=13)
    return qn


def queue_simu(qn, args, use_queue=True, n=None, t=None):
    edge_types = tuple(range(1, args.height+2))
    qn.initialize(edge_type=1)
    qn.clear_data()
    qn.start_collecting_data()
    qn.simulate(n=n, t=t)
    if use_queue:
        data = qn.get_queue_data(edge_type=edge_types, return_header=True)
    else:
        data = qn.get_agent_data(edge_type=edge_types, return_header=True)
    return data


def queue_show(qn, args):
    qn.g.new_vertex_property('pos')
    pos = {}
    h = 2
    for v in qn.g.nodes():
        if v == 0:
            pos[v] = [0, 0.4*args.height]
        elif v == 1:
            pos[v] = [0, 0.4*(args.height-1)]
        else:
            if args.weight != 1:
                n = args.weight
                if v > (pow(n, h)-1)/(n-1):
                    h += 1
                mid = (((n+1)*pow(n, h-1)-2)/(n-1)+1)/2
                pos[v] = [0.4*pow(n, args.height-h)*(v-mid), 0.4 * (args.height-h)]
            else:
                pos[v] = [0, 0.4*(args.height-v)]
    qn.g.set_pos(pos)
    qn.draw(figsize=(args.weight*args.height*5, len(qn.g.nodes())))


def data2csv(data_list, path):
    import pandas as pd
    data_pd = pd.DataFrame(data=data_list)
    data_name = path
    data_pd.to_csv(data_name, index=False)


def data2pickle(data_dic, path):
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(data_dic, f)


def simplfy_data(data_agent, limit_num=100, use_queue=True):
    if use_queue:
        simple_data = []
        for v in data_agent[0]:
            if v[Index.departure.value] > 0:
                simple_data.append([float('%.4f'%data) for data in v])
        simple_data = np.array(simple_data)
        q_ids = simple_data[:,-1]
        counter = Counter(q_ids)
        valid_index = []
        for q in set(q_ids):
            if counter[q] >= limit_num:
                valid_index.extend(np.where(q_ids == q)[0])
        simple_data = simple_data[valid_index]
        index = np.argsort(simple_data[:, Index.arrival.value])
        return np.array(simple_data)[index, :]
    else:
        simple_data = {}
        for k, v in data_agent[0].items():
            if v[-1, Index.departure.value] > 0:
                simple_data[k[1]] = v
        return simple_data


def adj_dict2mat(adja_list, num_nodes):
    mat = np.zeros((num_nodes, num_nodes))
    for k, v in adja_list.items():
        mat[k,v] =1
    return mat


def edge_adj2mat(g, valid_edges):
    """generate an adjcent matrix from the graph and constraint on valid edges"""
    edges = [e for e in g.edges() if e[0] != e[1]]
    mat = np.zeros([g.number_of_edges(), g.number_of_edges()])
    for edge_in in edges:
        edge_in_id = g.edge_index[edge_in]
        for edge_out in edges:
            edge_out_id = g.edge_index[edge_out]
            if edge_in[1] == edge_out[0] and edge_in_id in valid_edges and edge_out_id in valid_edges:
                mat[edge_in_id, edge_out_id] = 1
    return mat


def queue_data_dict2mat(data_dic):
    mat = []
    for k, v in data_dic.items():
        mat.extend(v)
    return mat


def gene_queue(args):
    qn = None
    file_head = ''
    if args.shape == 'tree':
        adja_list, edge_list = gen_tree(args.weight, args.height, entry)
        print(adja_list)
        print(edge_list)
        qn = graph2queue(adja_list, edge_list, args)
        file_head = 'weight_'+str(weight)+'_height_'+str(height)
    elif args.shape == 'pagerank':
        g = qt.generate_pagerank_graph(300, seed=13)
        qn = qt.QueueNetwork(g, seed=14)
        file_head = 'pagerank'
    return qn, file_head


if __name__=='__main__':
    args = parser.parse_args()
    weight = 1
    height = 5
    entry = (0, 1)
    args.weight = weight
    args.height = height
    args.shape = 'tree'
    qn, file_head = gene_queue(args)

    all_queues = qn.edge2queue
    # queue_show(qn, args)
    data = queue_simu(qn, args, use_queue=True, n=100000)

    data = simplfy_data(data, limit_num=100, use_queue=True)
    valid_queues = set(data[:, -1])

    print("length of valid queues: ", len(list(valid_queues)))
    print("length of valid data: ", len(data))
    print("length of edges: ", qn.g.number_of_edges())
    adjq_mat = edge_adj2mat(qn.g, valid_queues)
    # queue_mat = queue_data_dict2mat(data)
    data2csv(adjq_mat, file_head+'_adj.csv')

    data2csv(data, file_head+'_queue.csv')
