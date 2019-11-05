from torchvision.datasets.vision import VisionDataset
import os
import pickle
import re
import scipy.sparse as ss
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.distributed as dist
from threading import Thread
import multiprocessing as mp
from multiprocessing import Process
from multiprocessing.queues import SimpleQueue



class Simu(VisionDataset):
    def __init__(self, root, rank=0, world_size=1, train=True, transform=None, target_transform=None, download=True):
        super(Simu, self).__init__(root)
        # self.root = '/home/mag0a/mount/Projects/imagenet/logistic_regression/data'
        self.root = root
        self.base_folder = 'RCV1'
        self.file_name = ''

        self.train = train  # training set or test set

        self.rank = rank
        self.world_size = world_size
        self.split_dataset = False
        if self.world_size > 1:
            self.split_dataset = True
        if self.train:
            self.file_name = 'train_rank_' + str(rank) + '_ws_' + str(world_size) + '.pt'
        else:
            self.file_name = 'test' + '_ws_' + str(world_size) + '.pt'

        self.download = download

        if self.download:
            self.read_file()

        path = os.path.join(self.root, self.base_folder, self.file_name)
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        sparse_data = obj['data']
        self.rows, self.colums, self.each_row_num = sparse_data['rows'], sparse_data['colums'], sparse_data['each_row_num']
        self.max_num = max(self.each_row_num)+1
        # self.data = torch.sparse_coo_tensor(indices=torch.LongTensor([rows, colums]), values=sparse_data['data'], size=(max(rows)+1, max(colums)+1))
        self.data = ss.coo_matrix((sparse_data['data'], (self.rows, self.colums)), shape=(max(self.rows)+1, max(self.colums)+1))
        self.values = sparse_data['data']
        self.targets = obj['targets']
        self.mean, self.std = obj['mean'], obj['std']

        # self.transform = transform.transforms.append(transforms.Normalize(self.mean, self.std))
        self.target_transform = target_transform

        self.num_attr = self.data.shape[-1]
        self.num_targets = max(self.targets)+1

    def __getitem__(self, index):
        # tmp = ss.find(self.data.getrow(index))
        current_index = sum(self.each_row_num[0:index])
        rows = self.rows[current_index: current_index+self.each_row_num[index]]
        cols = self.colums[current_index: current_index+self.each_row_num[index]]
        vals = self.values[current_index: current_index+self.each_row_num[index]]
        length = self.each_row_num[index]
        img = torch.zeros([3, self.max_num])

        img[1, 0:length] += torch.tensor(cols).float()
        img[2, 0:length] += torch.tensor(vals)

        # rows = tmp[0]
        # columns = tmp[1]
        # value = tmp[2]
        # img = ss.coo_matrix((value, (rows, columns)), shape=(1, max(self.colums)+1))

        # img = torch.sparse_coo_tensor(indices=torch.LongTensor([rows, columns]), \
        #                               values=value, size=(1, max(self.colums)+1)).to_dense()[0]

        target = self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.data.shape[0]

    def get_length(self):
        return self.data.shape[0]

    def read_file(self):
        path = os.path.join(self.root, self.base_folder, self.file_name)
        if os.path.isfile(path):
            if self.rank == 0 or self.world_size == 1:
                print("data is prepared")
        else:
            train_path = os.path.join(self.root, self.base_folder, 'rcv1_train.binary')
            test_path = os.path.join(self.root, self.base_folder, 'rcv1_test.binary')

            train_data, train_targets = self.read_raw_file(train_path, self.split_dataset)
            print("read train raw file finished, prepare to dump into train.pt....")
            train = {'data': train_data, 'targets': train_targets, 'mean': 0, 'std': 0}
            with open(path, 'wb') as filehandler:
                pickle.dump(train, filehandler)
            print("dump into train", self.rank, ".pt finished, prepare to read test raw file....")

            sync = torch.zeros(1)
            if self.rank == 0 or self.world_size == 1:
                test_data, test_targets = self.read_raw_file(test_path)
                print("read test raw file finished, prepare to dump into test.pt....")
                test = {'data': test_data, 'targets': test_targets, 'mean': 0, 'std': 0}
                with open(path, 'wb') as filehandler:
                    pickle.dump(test, filehandler, protocol=4)
                print("dump into test.pt finished....")
                dist.all_reduce(sync)
            else:
                dist.all_reduce(sync)

    def read_raw_file(self, path, split_dataset=False):
        sparse_data = {'rows': [], 'colums': [], 'data': [], 'each_row_num': []}
        data = []
        targets = []
        rows = []
        colums = []
        each_row_num = []
        next_row = 0
        f = open(path)
        for i, x in enumerate(f.readlines()):
            if i % self.world_size == self.rank or split_dataset is False:
                tmp = re.split(":| ", x)
                targets.append(0 if int(tmp[0]) <= 0 else 1)
                rows.extend(next_row for _ in range(int((len(tmp) - 1) / 2)))
                colums.extend(int(tmp[j]) for j in range(1, len(tmp), 2))
                data.extend(float(tmp[j]) for j in range(2, len(tmp), 2))
                each_row_num.append(int((len(tmp) - 1) / 2))
                next_row += 1
        print("my rank is : ", self.rank, "length of data: ", next_row)
        # data = ss.coo_matrix((data, (rows, colums)), shape=(max(rows)+1, max(colums)+1))
        # data = torch.sparse_coo_tensor(indices=torch.LongTensor([rows, colums]), values=data, size=(max(rows)+1, max(colums)+1))
        sparse_data['rows'], sparse_data['colums'], sparse_data['data'], sparse_data['each_row_num'] = rows, colums, data, each_row_num
        return sparse_data, targets


    def get_mean(self, sparse_data):
        num_cols = sparse_data.shape[1]
        num_rows = sparse_data.shape[0]
        mean = np.ones(num_cols)*0.5
        # mean = [sum(data.getcol(i)).data[0]/num_rows for i in range(num_cols)]
        # for i in range(num_cols):
        #     if len(sparse_data.getcol(i).data)>0:
        #         mean.append(sum(sparse_data.getcol(i)).data[0]/num_rows)
        #     else:
        #         mean.append(0)
        return mean

    def get_std(self, sparse_data, mean):
        num_cols = sparse_data.shape[1]
        num_rows = sparse_data.shape[0]
        # mean2 = [sum(ss.find(sparse_data.getcol(i))[-1]**2)/num_rows for i in range(num_cols)]
        # for i in range(num_cols):
        #     if len(ss.find(sparse_data.getcol(i))[-1])>0:
        #         mean2.append(sum(ss.find(sparse_data.getcol(i))[-1]**2)/num_rows)
        #     else:
        #         mean2.append(1)
        # return [np.math.sqrt(i-j**2) for i,j in zip(mean2, mean)]
        return np.ones(num_cols)



if __name__=="__main__":
    transform = transforms.Compose(
        [transforms.ToTensor()])
    train_dataset = Simu(root='./data', train=True, download=True, transform=transform)
