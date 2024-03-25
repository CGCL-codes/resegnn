import argparse
import numpy as np
import os
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from history_dataset import HistoryDataset, HistoryDataset2, HistoryDataset3, HistoryDataset4


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


def hawk(alpha, beta, delta_time):
    return alpha * np.exp(-beta * delta_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='get history', usage='get_history.py [<args>] [-h | --help]')
    parser.add_argument('--dataset', default='ICEWS14', type=str)
    parser.add_argument('--method', default='time_decay', type=str)
    parser.add_argument('--num-workers', default=4, type=int)

    # hawk parameter
    parser.add_argument('--alpha', default=1, type=float)
    parser.add_argument('--beta', default=[1, 4], type=float,nargs=2) # for time_hawk_edge only
    parser.add_argument('--beta2', default=1, type=float) # for time_hawk_edge_2 only
    parser.add_argument('--k', default=20, type=int)# for time_hawk_edge_2 only
    parser.add_argument('--n', default=3, type=int)  # for time_hawk_edge_2 only

    args = parser.parse_args()

    data_path = '../data/{}'.format(args.dataset)
    all_data, all_times = load_all_quadruples('../data/{}'.format(args.dataset), 'train.txt', 'valid.txt', "test.txt")
    num_e, num_r = get_total_number('../data/{}'.format(args.dataset), 'stat.txt')
    save_dir_obj = '../data/{}/{}/history/'.format(args.dataset, args.method)
    mkdirs(save_dir_obj)
    raw_num_r = num_r
    num_r = num_r * 2
    time_span = (max(all_times) - min(all_times)) // args.k

    if args.method == 'time_hawk':
        history_dataset = HistoryDataset(args.dataset, args.method,args.alpha,args.beta)
        history_dataloader = DataLoader(dataset=history_dataset, batch_size=20, num_workers=args.num_workers,
                                        shuffle=False, prefetch_factor=5, pin_memory=True,persistent_workers=True)
        for tail_seq, rel_seq in tqdm(history_dataloader):
            pass

    if args.method == 'time_hawk_edge':
        history_dataset = HistoryDataset2(args.dataset, args.method,args.alpha,args.beta)
        history_dataloader = DataLoader(dataset=history_dataset, batch_size=10, num_workers=args.num_workers,
                                        shuffle=False, prefetch_factor=5, pin_memory=True,persistent_workers=True)
        for tail_seq, rel_seq in tqdm(history_dataloader):
            pass

    if args.method == 'time_hawk_edge_2':
        history_dataset = HistoryDataset3(args.dataset, args.method,args.alpha,args.beta2,args.k)
        history_dataloader = DataLoader(dataset=history_dataset, batch_size=20, num_workers=args.num_workers,
                                        shuffle=False, prefetch_factor=5, pin_memory=True,persistent_workers=True)
        for tail_seq, rel_seq in tqdm(history_dataloader):
            pass

    if args.method == 'frequency':
        history_dataset = HistoryDataset4(args.dataset, args.method,args.alpha,args.beta2,args.k)
        history_dataloader = DataLoader(dataset=history_dataset, batch_size=20, num_workers=args.num_workers,
                                        shuffle=False, prefetch_factor=5, pin_memory=True,persistent_workers=True)
        for tail_seq, rel_seq in tqdm(history_dataloader):
            pass

    if args.method == 'time_decay':
        for tim in tqdm(all_times):
            train_new_data = np.array([[quad[0], quad[1], quad[2], quad[3]] for quad in all_data if quad[3] < tim])
            if tim != all_times[0]:
                inverse_train_data = train_new_data[:, [2, 1, 0, 3]]
                inverse_train_data[:, 1] = inverse_train_data[:, 1] + raw_num_r
                train_new_data = np.concatenate([train_new_data, inverse_train_data])
                delta_time = tim - train_new_data[:, 3]
                dt = delta_time / time_span
                delta = 1 - np.arctan(dt) * 2 / np.pi
                train_new_data_unique, index = np.unique(train_new_data[:, :3], return_inverse=True, axis=0)
                # train_new_data = torch.from_numpy(train_new_data)

                # entity history
                row = train_new_data_unique[:, 0] * num_r + train_new_data_unique[:, 1]
                col = train_new_data_unique[:, 2]
                ent_semantic = np.zeros(train_new_data_unique.shape[0])
                for i, j in enumerate(index):
                    ent_semantic[j] += delta[i]
                tail_seq = sp.csr_matrix((ent_semantic, (row, col)), shape=(num_e * num_r, num_e))

                # relation history
                rel_row = train_new_data_unique[:, 0] * num_e + train_new_data_unique[:, 2]
                rel_col = train_new_data_unique[:, 1]
                rel_sematic = np.zeros(train_new_data_unique.shape[0])
                for i, j in enumerate(index):
                    rel_sematic[j] += delta[i]

                # print(np.sum(rel_sematic == 0))
                rel_seq = sp.csr_matrix((rel_sematic, (rel_row, rel_col)), shape=(num_e * num_e, num_r))
            else:
                tail_seq = sp.csr_matrix(([], ([], [])), shape=(num_e * num_r, num_e))
                rel_seq = sp.csr_matrix(([], ([], [])), shape=(num_e * num_e, num_r))
            sp.save_npz('../data/{}/{}/history/tail_history_{}.npz'.format(args.dataset, args.method, tim), tail_seq)
            sp.save_npz('../data/{}/{}/history/rel_history_{}.npz'.format(args.dataset, args.method, tim), rel_seq)
    elif args.method == 'once':
        for tim in tqdm(all_times):
            train_new_data = np.array([[quad[0], quad[1], quad[2], quad[3]] for quad in all_data if quad[3] < tim])
            if tim != all_times[0]:
                train_new_data = torch.from_numpy(train_new_data)
                inverse_train_data = train_new_data[:, [2, 1, 0, 3]]
                inverse_train_data[:, 1] = inverse_train_data[:, 1] + raw_num_r
                train_new_data = torch.cat([train_new_data, inverse_train_data])

                # entity history
                train_new_data = torch.unique(train_new_data[:, :3], sorted=False, dim=0)
                train_new_data = train_new_data.numpy()
                row = train_new_data[:, 0] * num_r + train_new_data[:, 1]
                col = train_new_data[:, 2]
                d = np.ones(len(row))
                tail_seq = sp.csr_matrix((d, (row, col)), shape=(num_e * num_r, num_e))

                # relation history
                rel_row = train_new_data[:, 0] * num_e + train_new_data[:, 2]
                rel_col = train_new_data[:, 1]
                rel_d = np.ones(len(rel_row))
                rel_seq = sp.csr_matrix((rel_d, (rel_row, rel_col)), shape=(num_e * num_e, num_r))
            else:
                tail_seq = sp.csr_matrix(([], ([], [])), shape=(num_e * num_r, num_e))
                rel_seq = sp.csr_matrix(([], ([], [])), shape=(num_e * num_e, num_r))
            sp.save_npz('../data/{}/{}/history/tail_history_{}.npz'.format(args.dataset, args.method, tim), tail_seq)
            sp.save_npz('../data/{}/{}/history/rel_history_{}.npz'.format(args.dataset, args.method, tim), rel_seq)
