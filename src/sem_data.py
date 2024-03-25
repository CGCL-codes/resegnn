import os
import pickle

import torch
import numpy as np
from torch.utils.data import Dataset, TensorDataset
import scipy.sparse as sp


def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    """
    将scipy的sparse matrix转换成torch的sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    # return torch.sparse.FloatTensor(indices, values, shape)
    return indices, values, shape


class SematicTrainDataset(Dataset):
    def __init__(self, train_list, train_history_len, num_nodes, num_rels, train_times, dataset, history_method,
                 use_cuda, gpu):
        self.train_list = train_list
        self.train_history_len = train_history_len
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.train_times = train_times
        self.filepath = '../data/{}/{}/history/'.format(dataset, history_method)
        self.use_cuda = use_cuda
        self.gpu = gpu

    def __getitem__(self, index):
        output = self.train_list[index:index + 1]
        if index - self.train_history_len < 0:
            start_history = 0
        else:
            start_history = index - self.train_history_len
        input_list = self.train_list[start_history: index]
        output = [torch.from_numpy(_).long() for _ in output]
        history_data = output[0]
        inverse_history_data = history_data[:, [2, 1, 0, 3]]
        inverse_history_data[:, 1] = inverse_history_data[:, 1] + self.num_rels
        history_data = torch.cat([history_data, inverse_history_data])
        history_data = history_data.numpy()
        all_tail_seq = sp.load_npz(os.path.join(self.filepath, 'tail_history_{}.npz'.format(self.train_times[index])))
        seq_idx = history_data[:, 0] * self.num_rels * 2 + history_data[:, 1]
        tail_seq = scipy_sparse_mat_to_torch_sparse_tensor(all_tail_seq[seq_idx])
        all_rel_seq = sp.load_npz(os.path.join(self.filepath, 'rel_history_{}.npz'.format(self.train_times[index])))
        rel_seq_idx = history_data[:, 0] * self.num_nodes + history_data[:, 2]
        rel_seq = scipy_sparse_mat_to_torch_sparse_tensor(all_rel_seq[rel_seq_idx])
        return input_list, start_history, output, tail_seq, rel_seq

    def __len__(self):
        return len(self.train_list)


class SematicTestDataset(Dataset):
    def __init__(self, test_list, test_history_len, num_nodes, num_rels, time_list, dataset,
                 history_method, multi_step, history_time_nogt, use_cuda, gpu):
        self.test_list = test_list
        self.test_history_len = test_history_len
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.time_list = time_list
        self.filepath = '../data/{}/{}/history/'.format(dataset, history_method)
        self.multi_step = multi_step
        self.history_time_nogt = history_time_nogt
        self.use_cuda = use_cuda
        self.gpu = gpu

    def __getitem__(self, index):
        test_snap = self.test_list[index]
        test_triples_input = torch.LongTensor(test_snap)
        history_data = test_triples_input
        inverse_history_data = history_data[:, [2, 1, 0, 3]]
        inverse_history_data[:, 1] = inverse_history_data[:, 1] + self.num_rels
        history_data = torch.cat([history_data, inverse_history_data])
        history_data = history_data.numpy()
        if self.multi_step:
            history_index = self.history_time_nogt
        else:
            history_index = self.time_list[index]
        all_tail_seq = sp.load_npz(os.path.join(self.filepath, 'tail_history_{}.npz'.format(history_index)))
        seq_idx = history_data[:, 0] * self.num_rels * 2 + history_data[:, 1]
        tail_seq = scipy_sparse_mat_to_torch_sparse_tensor(all_tail_seq[seq_idx])
        all_rel_seq = sp.load_npz(os.path.join(self.filepath, 'rel_history_{}.npz'.format(history_index)))
        rel_seq_idx = history_data[:, 0] * self.num_nodes + history_data[:, 2]
        rel_seq = scipy_sparse_mat_to_torch_sparse_tensor(all_rel_seq[rel_seq_idx])
        return test_snap, test_triples_input, tail_seq, rel_seq

    def __len__(self):
        return len(self.test_list)

def mycollate(batchs):
    result = []
    for i in range(len(batchs[0])):
        result.append([batch[i] for batch in batchs])
    return result