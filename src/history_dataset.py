import argparse
import numpy as np
import os
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def load_quadruples(inPath, fileName, fileName2=None):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)
    if fileName2 is not None:
        with open(os.path.join(inPath, fileName2), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)
    times = list(times)
    times.sort()

    return np.asarray(quadrupleList), np.asarray(times)


def load_all_quadruples(inPath, fileName, fileName2=None, fileName3=None):
    quadrupleList = []
    times = set()
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)
    with open(os.path.join(inPath, fileName2), 'r') as fr:
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)
    with open(os.path.join(inPath, fileName3), 'r') as fr:
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)
    times = list(times)
    times.sort()

    return np.asarray(quadrupleList), np.asarray(times)


def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])


def get_data_with_t(data, tim):
    triples = [[quad[0], quad[1], quad[2]] for quad in data if quad[3] == tim]
    return np.array(triples)


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def hawk(alpha, beta, delta_time, lambdas=0):
    return lambdas + alpha * np.exp(-beta * delta_time)


class HistoryDataset(Dataset):
    def __init__(self, dataset, method, alpha, beta_args):
        self.dataset = dataset
        self.method = method
        self.alpha = alpha
        self.beta_args = beta_args
        self.data_path = '../data/{}'.format(dataset)
        self.all_data, self.all_times = load_all_quadruples('../data/{}'.format(dataset), 'train.txt', 'valid.txt',
                                                            "test.txt")
        self.num_e, self.num_r = get_total_number('../data/{}'.format(dataset), 'stat.txt')
        method = self.method + '-alpha{}-beta{}_{}'.format(self.alpha, self.beta_args[0], self.beta_args[1])
        self.save_dir_obj = '../data/{}/{}/history/'.format(dataset, method)
        if not os.path.exists(self.save_dir_obj):
            mkdirs(self.save_dir_obj)
        self.raw_num_r = self.num_r
        self.num_r = self.num_r * 2
        # 数据集采集间隔
        time_unit = {'GDELT': 1, 'ICEWS14': 4 * 24, 'ICEWS18': 4 * 24, 'ICEWS05-15': 4 * 24, 'WIKI': 4 * 24 * 365,
                     'YAGO': 4 * 24 * 365}
        # 数据集中的最小时间间隔
        self.time_tick = {'GDELT': 15, 'ICEWS14': 24, 'ICEWS18': 24, 'ICEWS05-15': 1, 'WIKI': 1, 'YAGO': 1}
        self.max_time = {'GDELT': 44625, 'ICEWS14': 8736, 'ICEWS18': 7272, 'ICEWS05-15': 4016, 'WIKI': 231, 'YAGO': 188}
        self.beta = beta_args[0] + np.log(time_unit[dataset]) / np.log(beta_args[1])

    def __len__(self):
        return len(self.all_times)

    def __getitem__(self, index):
        tim = self.all_times[index]
        train_new_data = np.array([[quad[0], quad[1], quad[2], quad[3]] for quad in self.all_data if quad[3] < tim])
        if tim != self.all_times[0]:
            inverse_train_data = train_new_data[:, [2, 1, 0, 3]]
            inverse_train_data[:, 1] = inverse_train_data[:, 1] + self.raw_num_r
            train_new_data = np.concatenate([train_new_data, inverse_train_data])
            dt = (tim - train_new_data[:, 3]) / self.time_tick[self.dataset]
            delta = hawk(self.alpha, self.beta, dt)
            train_new_data_unique, index = np.unique(train_new_data[:, :3], return_inverse=True, axis=0)
            # train_new_data = torch.from_numpy(train_new_data)

            # entity triple history
            row = train_new_data_unique[:, 0] * self.num_r + train_new_data_unique[:, 1]
            col = train_new_data_unique[:, 2]
            ent_semantic = np.zeros(train_new_data_unique.shape[0])
            for i, j in enumerate(index):
                if ent_semantic[j] == 0:
                    ent_semantic[j] = 1
                ent_semantic[j] += delta[i]
            tail_seq = sp.csr_matrix((ent_semantic, (row, col)), shape=(self.num_e * self.num_r, self.num_e))

            # entity history
            # row2 = train_new_data_unique[:, 0]
            # col2 = train_new_data_unique[:, 1]
            # ent_semantic = np.zeros(train_new_data_unique.shape[0])

            # relation triple history
            rel_row = train_new_data_unique[:, 0] * self.num_e + train_new_data_unique[:, 2]
            rel_col = train_new_data_unique[:, 1]
            rel_sematic = np.zeros(train_new_data_unique.shape[0])
            for i, j in enumerate(index):
                if rel_sematic[j] == 0:
                    rel_sematic[j] = 1
                rel_sematic[j] += delta[i]
            # print(np.sum(rel_sematic == 0))
            rel_seq = sp.csr_matrix((rel_sematic, (rel_row, rel_col)), shape=(self.num_e * self.num_e, self.num_r))
        else:
            tail_seq = sp.csr_matrix(([], ([], [])), shape=(self.num_e * self.num_r, self.num_e))
            rel_seq = sp.csr_matrix(([], ([], [])), shape=(self.num_e * self.num_e, self.num_r))

        sp.save_npz(self.save_dir_obj + 'tail_history_{}.npz'.format(tim), tail_seq)
        sp.save_npz(self.save_dir_obj + 'rel_history_{}.npz'.format(tim), rel_seq)
        return [], []


class HistoryDataset2(Dataset):
    def __init__(self, dataset, method, alpha, beta_args):
        self.dataset = dataset
        self.method = method
        self.alpha = alpha
        self.beta_args = beta_args
        self.data_path = '../data/{}'.format(dataset)
        self.all_data, self.all_times = load_all_quadruples('../data/{}'.format(dataset), 'train.txt', 'valid.txt',
                                                            "test.txt")
        self.num_e, self.num_r = get_total_number('../data/{}'.format(dataset), 'stat.txt')
        method = self.method + '-alpha{}-beta{}_{}'.format(self.alpha, self.beta_args[0], self.beta_args[1])
        self.save_dir_obj = '../data/{}/{}/history/'.format(dataset, method)
        mkdirs(self.save_dir_obj)
        self.raw_num_r = self.num_r
        self.num_r = self.num_r * 2
        # set 15min as 1
        self.time_unit = {'GDELT': 1, 'ICEWS14': 4 * 24, 'ICEWS18': 4 * 24, 'ICEWS05-15': 4 * 24, 'WIKI': 4 * 24 * 365,
                          'YAGO': 4 * 24 * 365}
        self.time_tick = {'GDELT': 15, 'ICEWS14': 24, 'ICEWS18': 24, 'ICEWS05-15': 1, 'WIKI': 1, 'YAGO': 1}
        self.beta = beta_args[0] + np.log(self.time_unit[dataset]) / np.log(beta_args[1])

    def __len__(self):
        return len(self.all_times)

    def __getitem__(self, index):
        tim = self.all_times[index]
        if os.path.exists(self.save_dir_obj + 'tail_history_{}.npz'.format(tim)) \
                and os.path.exists(self.save_dir_obj + 'tail_edge_history_{}.npz'.format(tim)):
            return [], []
        train_new_data = np.array([[quad[0], quad[1], quad[2], quad[3]] for quad in self.all_data if quad[3] < tim])
        if tim != self.all_times[0]:
            inverse_train_data = train_new_data[:, [2, 1, 0, 3]]
            inverse_train_data[:, 1] = inverse_train_data[:, 1] + self.raw_num_r
            train_new_data = np.concatenate([train_new_data, inverse_train_data])
            dt = (tim - train_new_data[:, 3]) / self.time_tick[self.dataset]
            delta = hawk(self.alpha, self.beta, dt)
            train_new_data_unique, index = np.unique(train_new_data[:, :3], return_inverse=True, axis=0)
            # train_new_data = torch.from_numpy(train_new_data)
            if not os.path.exists(self.save_dir_obj + 'tail_history_{}.npz'.format(tim)) or \
                    not os.path.exists(self.save_dir_obj + 'rel_history_{}.npz'.format(tim)):
                # entity triple history
                row = train_new_data_unique[:, 0] * self.num_r + train_new_data_unique[:, 1]
                col = train_new_data_unique[:, 2]
                ent_semantic = np.zeros(train_new_data_unique.shape[0])
                for i, j in enumerate(index):
                    if ent_semantic[j] == 0:
                        ent_semantic[j] = 1
                    ent_semantic[j] += delta[i]
                tail_seq = sp.csr_matrix((ent_semantic, (row, col)), shape=(self.num_e * self.num_r, self.num_e))

                # relation triple history
                rel_row = train_new_data_unique[:, 0] * self.num_e + train_new_data_unique[:, 2]
                rel_col = train_new_data_unique[:, 1]
                rel_sematic = np.zeros(train_new_data_unique.shape[0])
                for i, j in enumerate(index):
                    if rel_sematic[j] == 0:
                        rel_sematic[j] = 1
                    rel_sematic[j] += delta[i]
                # print(np.sum(rel_sematic == 0))
                rel_seq = sp.csr_matrix((rel_sematic, (rel_row, rel_col)), shape=(self.num_e * self.num_e, self.num_r))
                sp.save_npz(self.save_dir_obj + 'tail_history_{}.npz'.format(tim), tail_seq)
                sp.save_npz(self.save_dir_obj + 'rel_history_{}.npz'.format(tim), rel_seq)

            if not os.path.exists(self.save_dir_obj + 'tail_edge_history_{}.npz'.format(tim)) \
                    or not os.path.exists(self.save_dir_obj + 'rel_edge_history_{}.npz'.format(tim)):
                # object-subject history&relation-subject history
                tail_os_seq = torch.zeros(self.num_e, self.num_e)
                tail_rs_seq = torch.zeros(self.num_r, self.num_e)
                rel_hr_seq = torch.zeros(self.num_e, self.num_r)
                for i in range(train_new_data.shape[0]):
                    tail_os_seq[train_new_data[:, 0][i], train_new_data[:, 2][i]] += delta[i]
                    tail_rs_seq[train_new_data[:, 1][i], train_new_data[:, 2][i]] += delta[i]
                    rel_hr_seq[train_new_data[:, 0][i], train_new_data[:, 1][i]] += delta[i]

                top_select = 3
                values_os, idx_os = torch.topk(tail_os_seq, k=top_select, dim=-1)
                values_rs, idx_rs = torch.topk(tail_rs_seq, k=top_select, dim=-1)
                values_hr, idx_hr = torch.topk(rel_hr_seq, k=top_select, dim=-1)
                tail_os_his = 0
                tail_rs_his = 0
                rel_hr_his = 0

                rows_os = np.arange(self.num_e * self.num_r)
                rows_rs = np.arange(self.num_e * self.num_r)
                rows_hr = np.arange(self.num_e * self.num_e)
                for i in range(top_select):
                    cols_os = np.repeat(idx_os[:, i], self.num_r)
                    values_sp_os = np.repeat(values_os[:, i], self.num_r)
                    tail_os_his += sp.csr_matrix((values_sp_os, (rows_os, cols_os)),
                                                 shape=(self.num_e * self.num_r, self.num_e))

                    cols_rs = np.tile(idx_rs[:, i], self.num_e)
                    values_sp_rs = np.tile(values_rs[:, i], self.num_e)
                    tail_rs_his += sp.csr_matrix((values_sp_rs, (rows_rs, cols_rs)),
                                                 shape=(self.num_e * self.num_r, self.num_e))

                    cols_hr = np.repeat(idx_hr[:, i], self.num_e)
                    values_sp_hr = np.repeat(values_hr[:, i], self.num_e)
                    rel_hr_his += sp.csr_matrix((values_sp_hr, (rows_hr, cols_hr)),
                                                shape=(self.num_e * self.num_e, self.num_r))
                sp.save_npz(self.save_dir_obj + 'tail_edge_history_{}.npz'.format(tim), tail_os_his + tail_rs_his)
                sp.save_npz(self.save_dir_obj + 'rel_edge_history_{}.npz'.format(tim), rel_hr_his)
        else:
            tail_seq = sp.csr_matrix(([], ([], [])), shape=(self.num_e * self.num_r, self.num_e))
            rel_seq = sp.csr_matrix(([], ([], [])), shape=(self.num_e * self.num_e, self.num_r))
            tail_os_his = sp.csr_matrix(([], ([], [])), shape=(self.num_e * self.num_r, self.num_e))
            tail_rs_his = sp.csr_matrix(([], ([], [])), shape=(self.num_e * self.num_r, self.num_e))
            rel_hr_his = sp.csr_matrix(([], ([], [])), shape=(self.num_e * self.num_e, self.num_r))
            sp.save_npz(self.save_dir_obj + 'tail_history_{}.npz'.format(tim), tail_seq)
            sp.save_npz(self.save_dir_obj + 'rel_history_{}.npz'.format(tim), rel_seq)
            sp.save_npz(self.save_dir_obj + 'tail_edge_history_{}.npz'.format(tim), tail_os_his + tail_rs_his)
            sp.save_npz(self.save_dir_obj + 'rel_edge_history_{}.npz'.format(tim), rel_hr_his)
        return [], []


def hawk_with_ground(alpha, beta, delta_time, lambdas=0):
    return np.where(alpha * np.exp(-beta * delta_time) < 0.001, 0.001,alpha * np.exp(-beta * delta_time))


class HistoryDataset3(Dataset):
    """time hawkes with min value and fixed len"""
    def __init__(self, dataset, method, alpha, beta,k):
        self.dataset = dataset
        self.method = method
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.data_path = '../data/{}'.format(dataset)
        self.all_data, self.all_times = load_all_quadruples('../data/{}'.format(dataset), 'train.txt', 'valid.txt',
                                                            "test.txt")
        self.num_e, self.num_r = get_total_number('../data/{}'.format(dataset), 'stat.txt')
        method = self.method + '-alpha{}-beta{}-k{}'.format(self.alpha, self.beta,self.k)
        self.save_dir_obj = '../data/{}/{}/history/'.format(dataset, method)
        mkdirs(self.save_dir_obj)
        self.raw_num_r = self.num_r
        self.num_r = self.num_r * 2
        # set 15min as 1
        self.time_unit = {'GDELT': 1, 'ICEWS14': 4 * 24, 'ICEWS18': 4 * 24, 'ICEWS05-15': 4 * 24, 'WIKI': 4 * 24 * 365,
                          'YAGO': 4 * 24 * 365}
        self.time_tick = {'GDELT': 15, 'ICEWS14': 24, 'ICEWS18': 24, 'ICEWS05-15': 1, 'WIKI': 1, 'YAGO': 1}
        self.max_time = {'GDELT': 44625, 'ICEWS14': 8736, 'ICEWS18': 7272, 'ICEWS05-15': 4016, 'WIKI': 231, 'YAGO': 188}
        self.time_span = (max(self.all_times) - min(self.all_times)) // self.k

    def __len__(self):
        return len(self.all_times)

    def __getitem__(self, index):
        tim = self.all_times[index]
        if os.path.exists(self.save_dir_obj + 'tail_history_{}.npz'.format(tim)) \
                and os.path.exists(self.save_dir_obj + 'tail_edge_history_{}.npz'.format(tim)):
            return [], []
        train_new_data = np.array([[quad[0], quad[1], quad[2], quad[3]] for quad in self.all_data if quad[3] < tim])
        if tim != self.all_times[0]:
            inverse_train_data = train_new_data[:, [2, 1, 0, 3]]
            inverse_train_data[:, 1] = inverse_train_data[:, 1] + self.raw_num_r
            train_new_data = np.concatenate([train_new_data, inverse_train_data])
            # dt = (tim - train_new_data[:, 3]) / self.time_tick[self.dataset]
            dt = (tim - train_new_data[:, 3]) / self.time_span
            # delta = hawk_with_ground(self.alpha,self.beta[0], dt)
            delta = hawk_with_ground(self.alpha, self.beta, dt)
            train_new_data_unique, index = np.unique(train_new_data[:, :3], return_inverse=True, axis=0)
            # train_new_data = torch.from_numpy(train_new_data)
            if not os.path.exists(self.save_dir_obj + 'tail_history_{}.npz'.format(tim)) or \
                    not os.path.exists(self.save_dir_obj + 'rel_history_{}.npz'.format(tim)):
                # entity triple history
                row = train_new_data_unique[:, 0] * self.num_r + train_new_data_unique[:, 1]
                col = train_new_data_unique[:, 2]
                ent_semantic = np.zeros(train_new_data_unique.shape[0])
                for i, j in enumerate(index):
                    if ent_semantic[j] == 0:
                        ent_semantic[j] = 1
                    ent_semantic[j] += delta[i]
                tail_seq = sp.csr_matrix((ent_semantic, (row, col)), shape=(self.num_e * self.num_r, self.num_e))

                # relation triple history
                rel_row = train_new_data_unique[:, 0] * self.num_e + train_new_data_unique[:, 2]
                rel_col = train_new_data_unique[:, 1]
                rel_sematic = np.zeros(train_new_data_unique.shape[0])
                for i, j in enumerate(index):
                    if rel_sematic[j] == 0:
                        rel_sematic[j] = 1
                    rel_sematic[j] += delta[i]
                # print(np.sum(rel_sematic == 0))
                rel_seq = sp.csr_matrix((rel_sematic, (rel_row, rel_col)), shape=(self.num_e * self.num_e, self.num_r))
                sp.save_npz(self.save_dir_obj + 'tail_history_{}.npz'.format(tim), tail_seq)
                sp.save_npz(self.save_dir_obj + 'rel_history_{}.npz'.format(tim), rel_seq)

            if not os.path.exists(self.save_dir_obj + 'tail_edge_history_{}.npz'.format(tim)) \
                    or not os.path.exists(self.save_dir_obj + 'rel_edge_history_{}.npz'.format(tim)):
                # object-subject history&relation-subject history
                tail_os_seq = torch.zeros(self.num_e, self.num_e)
                tail_rs_seq = torch.zeros(self.num_r, self.num_e)
                rel_hr_seq = torch.zeros(self.num_e, self.num_r)
                for i in range(train_new_data.shape[0]):
                    tail_os_seq[train_new_data[:, 0][i], train_new_data[:, 2][i]] += delta[i]
                    tail_rs_seq[train_new_data[:, 1][i], train_new_data[:, 2][i]] += delta[i]
                    rel_hr_seq[train_new_data[:, 0][i], train_new_data[:, 1][i]] += delta[i]

                top_select = 3
                values_os, idx_os = torch.topk(tail_os_seq, k=top_select, dim=-1)
                values_rs, idx_rs = torch.topk(tail_rs_seq, k=top_select, dim=-1)
                values_hr, idx_hr = torch.topk(rel_hr_seq, k=top_select, dim=-1)
                tail_os_his = 0
                tail_rs_his = 0
                rel_hr_his = 0

                rows_os = np.arange(self.num_e * self.num_r)
                rows_rs = np.arange(self.num_e * self.num_r)
                rows_hr = np.arange(self.num_e * self.num_e)
                for i in range(top_select):
                    cols_os = np.repeat(idx_os[:, i], self.num_r)
                    values_sp_os = np.repeat(values_os[:, i], self.num_r)
                    tail_os_his += sp.csr_matrix((values_sp_os, (rows_os, cols_os)),
                                                 shape=(self.num_e * self.num_r, self.num_e))

                    cols_rs = np.tile(idx_rs[:, i], self.num_e)
                    values_sp_rs = np.tile(values_rs[:, i], self.num_e)
                    tail_rs_his += sp.csr_matrix((values_sp_rs, (rows_rs, cols_rs)),
                                                 shape=(self.num_e * self.num_r, self.num_e))

                    cols_hr = np.repeat(idx_hr[:, i], self.num_e)
                    values_sp_hr = np.repeat(values_hr[:, i], self.num_e)
                    rel_hr_his += sp.csr_matrix((values_sp_hr, (rows_hr, cols_hr)),
                                                shape=(self.num_e * self.num_e, self.num_r))
                sp.save_npz(self.save_dir_obj + 'tail_edge_history_{}.npz'.format(tim), tail_os_his + tail_rs_his)
                sp.save_npz(self.save_dir_obj + 'rel_edge_history_{}.npz'.format(tim), rel_hr_his)
        else:
            tail_seq = sp.csr_matrix(([], ([], [])), shape=(self.num_e * self.num_r, self.num_e))
            rel_seq = sp.csr_matrix(([], ([], [])), shape=(self.num_e * self.num_e, self.num_r))
            tail_os_his = sp.csr_matrix(([], ([], [])), shape=(self.num_e * self.num_r, self.num_e))
            tail_rs_his = sp.csr_matrix(([], ([], [])), shape=(self.num_e * self.num_r, self.num_e))
            rel_hr_his = sp.csr_matrix(([], ([], [])), shape=(self.num_e * self.num_e, self.num_r))
            sp.save_npz(self.save_dir_obj + 'tail_history_{}.npz'.format(tim), tail_seq)
            sp.save_npz(self.save_dir_obj + 'rel_history_{}.npz'.format(tim), rel_seq)
            sp.save_npz(self.save_dir_obj + 'tail_edge_history_{}.npz'.format(tim), tail_os_his + tail_rs_his)
            sp.save_npz(self.save_dir_obj + 'rel_edge_history_{}.npz'.format(tim), rel_hr_his)
        return [], []

class HistoryDataset4(Dataset):
    """time hawkes with min value and fixed len"""
    def __init__(self, dataset, method, alpha, beta,k):
        self.dataset = dataset
        self.method = method
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.data_path = '../data/{}'.format(dataset)
        self.all_data, self.all_times = load_all_quadruples('../data/{}'.format(dataset), 'train.txt', 'valid.txt',
                                                            "test.txt")
        self.num_e, self.num_r = get_total_number('../data/{}'.format(dataset), 'stat.txt')
        method = self.method + '-alpha{}-beta{}-k{}'.format(self.alpha, self.beta,self.k)
        self.save_dir_obj = '../data/{}/{}/history/'.format(dataset, method)
        mkdirs(self.save_dir_obj)
        self.raw_num_r = self.num_r
        self.num_r = self.num_r * 2


    def __len__(self):
        return len(self.all_times)

    def __getitem__(self, index):
        tim = self.all_times[index]
        if os.path.exists(self.save_dir_obj + 'tail_history_{}.npz'.format(tim)) \
                and os.path.exists(self.save_dir_obj + 'tail_edge_history_{}.npz'.format(tim)):
            return [], []
        train_new_data = np.array([[quad[0], quad[1], quad[2], quad[3]] for quad in self.all_data if quad[3] < tim])
        if tim != self.all_times[0]:
            inverse_train_data = train_new_data[:, [2, 1, 0, 3]]
            inverse_train_data[:, 1] = inverse_train_data[:, 1] + self.raw_num_r
            train_new_data = np.concatenate([train_new_data, inverse_train_data])
            train_new_data_unique, index = np.unique(train_new_data[:, :3], return_inverse=True, axis=0)
            # train_new_data = torch.from_numpy(train_new_data)
            if not os.path.exists(self.save_dir_obj + 'tail_history_{}.npz'.format(tim)) or \
                    not os.path.exists(self.save_dir_obj + 'rel_history_{}.npz'.format(tim)):
                # entity triple history
                row = train_new_data_unique[:, 0] * self.num_r + train_new_data_unique[:, 1]
                col = train_new_data_unique[:, 2]
                ent_semantic = np.zeros(train_new_data_unique.shape[0])
                for i, j in enumerate(index):
                    ent_semantic[j] += 1
                tail_seq = sp.csr_matrix((ent_semantic, (row, col)), shape=(self.num_e * self.num_r, self.num_e))

                # relation triple history
                rel_row = train_new_data_unique[:, 0] * self.num_e + train_new_data_unique[:, 2]
                rel_col = train_new_data_unique[:, 1]
                rel_sematic = np.zeros(train_new_data_unique.shape[0])
                for i, j in enumerate(index):
                    rel_sematic[j] += 1
                # print(np.sum(rel_sematic == 0))
                rel_seq = sp.csr_matrix((rel_sematic, (rel_row, rel_col)), shape=(self.num_e * self.num_e, self.num_r))
                sp.save_npz(self.save_dir_obj + 'tail_history_{}.npz'.format(tim), tail_seq)
                sp.save_npz(self.save_dir_obj + 'rel_history_{}.npz'.format(tim), rel_seq)

        else:
            tail_seq = sp.csr_matrix(([], ([], [])), shape=(self.num_e * self.num_r, self.num_e))
            rel_seq = sp.csr_matrix(([], ([], [])), shape=(self.num_e * self.num_e, self.num_r))
            sp.save_npz(self.save_dir_obj + 'tail_history_{}.npz'.format(tim), tail_seq)
            sp.save_npz(self.save_dir_obj + 'rel_history_{}.npz'.format(tim), rel_seq)
        return [], []