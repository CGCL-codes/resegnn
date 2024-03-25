import argparse
import itertools
import os
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import scipy.sparse as sp
import torch.nn.modules.rnn

from sem_data_edge import SematicTestDataset2, SematicTrainDataset2,mycollate

sys.path.append("..")
from rgcn import utils
from rgcn.utils import build_sub_graph
from src.model import RESEGNN
from src.hyperparameter_range import *
from rgcn.knowledge_graph import _read_triplets_as_list
# from sem_data import SematicTrainDataset, mycollate, SematicTestDataset


def test(model, history_list, test_list, num_rels, num_nodes, use_cuda, all_ans_list, all_ans_r_list, model_name,
         time_list, history_time_nogt, static_graph, mode):
    """
    :param model: model used to test
    :param history_list:    all input history snap shot list, not include output label train list or valid list
    :param test_list:   test triple snap shot list
    :param num_rels:    number of relations
    :param num_nodes:   number of nodes
    :param use_cuda:
    :param all_ans_list:     dict used to calculate filter mrr (key and value are all int variable not tensor)
    :param all_ans_r_list:     dict used to calculate filter mrr (key and value are all int variable not tensor)
    :param model_name:
    :param mode
    :return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r
    """
    ranks_raw, ranks_filter, mrr_raw_list, mrr_filter_list = [], [], [], []
    ranks_raw_r, ranks_filter_r, mrr_raw_list_r, mrr_filter_list_r = [], [], [], []
    all_entity_sematic = None
    all_rel_sematic = None

    if mode == "test":
        # test mode: load parameter form file
        if use_cuda:
            checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
        else:
            checkpoint = torch.load(model_name, map_location=torch.device('cpu'))
        print("Load Model name: {}. Using best epoch : {}"
              .format(model_name, checkpoint['epoch']))  # use best stat checkpoint
        print("----------------------------------------start testing----------------------------------------")
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    # do not have inverse relation in test input
    input_list = [snap for snap in history_list[-args.test_history_len:]]

    test_data = SematicTestDataset2(test_list, args.test_history_len, num_nodes, num_rels, time_list,
                                   args.dataset, args.history_method, args.multi_step, history_time_nogt, use_cuda,
                                   args.gpu)
    test_dataloader = DataLoader(dataset=test_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
                                 collate_fn=mycollate, prefetch_factor=10, pin_memory=True,persistent_workers=True)

    time_idx = 0
    for test_snap, test_triples_input, tail_seq, rel_seq,tail_edge_seq, rel_edge_seq in tqdm(test_dataloader):
        for test_snap_i, test_triples_input_i, tail_seq_i, rel_seq_i,tail_edge_seq_i, rel_edge_seq_i \
                in zip(test_snap, test_triples_input, tail_seq, rel_seq,tail_edge_seq, rel_edge_seq):
            history_glist = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu) for snap in input_list]
            # if use_cuda:
            #     tail_seq_i = tail_seq_i.cuda()
            #     rel_seq_i = rel_seq_i.cuda()
            tail_seq_i = torch.sparse.FloatTensor(*tail_seq_i)
            rel_seq_i = torch.sparse.FloatTensor(*rel_seq_i)
            if 'edge' in args.history_method:
                tail_edge_seq_i = torch.sparse.FloatTensor(*tail_edge_seq_i)
                rel_edge_seq_i = torch.sparse.FloatTensor(*rel_edge_seq_i)
            if use_cuda:
                test_triples_input_i = test_triples_input_i.cuda()
                tail_seq_i = tail_seq_i.cuda()
                rel_seq_i = rel_seq_i.cuda()
                if 'edge' in args.history_method:
                    tail_edge_seq_i = tail_edge_seq_i.cuda()
                    rel_edge_seq_i = rel_edge_seq_i.cuda()
            tail_seq_i = tail_seq_i.to_dense()
            rel_seq_i = rel_seq_i.to_dense()
            if 'edge' in args.history_method:
                tail_edge_seq_i = tail_edge_seq_i.to_dense()
                rel_edge_seq_i = rel_edge_seq_i.to_dense()

            start_history = len(history_list) - args.test_history_len
            test_triples, final_score, final_r_score = \
                model.predict(history_glist, start_history, num_rels, static_graph, test_triples_input_i, tail_seq_i, rel_seq_i,tail_edge_seq_i,rel_edge_seq_i)

            mrr_filter_snap_r, mrr_snap_r, rank_raw_r, rank_filter_r \
                = utils.get_total_rank(test_triples, final_r_score, all_ans_r_list[time_idx], eval_bz=1000, rel_predict=1)
            mrr_filter_snap, mrr_snap, rank_raw, rank_filter \
                = utils.get_total_rank(test_triples, final_score, all_ans_list[time_idx], eval_bz=1000, rel_predict=0)

            # used to global statistic
            ranks_raw.append(rank_raw)
            ranks_filter.append(rank_filter)
            # used to show slide results
            mrr_raw_list.append(mrr_snap)
            mrr_filter_list.append(mrr_filter_snap)

            # relation rank
            ranks_raw_r.append(rank_raw_r)
            ranks_filter_r.append(rank_filter_r)
            mrr_raw_list_r.append(mrr_snap_r)
            mrr_filter_list_r.append(mrr_filter_snap_r)

            # reconstruct history graph list
            if args.multi_step:
                if not args.relation_evaluation:
                    predicted_snap = utils.construct_snap(test_triples, num_nodes, num_rels, final_score, args.topk)
                else:
                    predicted_snap = utils.construct_snap_r(test_triples, num_nodes, num_rels, final_r_score, args.topk)
                # 长度等于零认为不用多步预测
                if len(predicted_snap):
                    input_list.pop(0)
                    input_list.append(predicted_snap)
            else:
                input_list.pop(0)
                input_list.append(test_snap_i)
            time_idx += 1

    print("|{:^18}|{:^9}|{:^9}|{:^9}|{:^9}| |{:^18}|{:^9}|{:^9}|{:^9}|{:^9}|"
          .format('entity metrics', 'MRR (%)', 'hit@1(%)', 'hit@3(%)', 'hit@10(%)', 'relation metrics', 'MRR (%)',
                  'hit@1(%)', 'hit@3(%)', 'hit@10(%)'))
    print("|{:-^18}|{:-^9}|{:-^9}|{:-^9}|{:-^9}| |{:-^18}|{:-^9}|{:-^9}|{:-^9}|{:-^9}|"
          .format('', '', '', '', '', '', '', '', '', ''))
    mrr_raw, hits_value_raw = utils.stat_ranks(ranks_raw)
    mrr_filter, hits_value_filtered = utils.stat_ranks(ranks_filter)

    mrr_raw_r, hits_value_raw_r = utils.stat_ranks(ranks_raw_r)
    mrr_filter_r, hits_value_filtered_r = utils.stat_ranks(ranks_filter_r)
    hits_value_raw = [hits_value.item() * 100 for hits_value in hits_value_raw]
    hits_value_raw_r = [hits_value.item() * 100 for hits_value in hits_value_raw_r]
    hits_value_filtered = [hits_value.item() * 100 for hits_value in hits_value_filtered]
    hits_value_filtered_r = [hits_value.item() * 100 for hits_value in hits_value_filtered_r]
    print("|{:^18}|{:^9.4f}|{:^9.4f}|{:^9.4f}|{:^9.4f}| |{:^18}|{:^9.4f}|{:^9.4f}|{:^9.4f}|{:^9.4f}|"
          .format('raw', mrr_raw.item() * 100, hits_value_raw[0], hits_value_raw[1], hits_value_raw[2],
                  'raw', mrr_raw_r.item() * 100, hits_value_raw_r[0], hits_value_raw_r[1], hits_value_raw_r[2]))
    print("|{:^18}|{:^9.4f}|{:^9.4f}|{:^9.4f}|{:^9.4f}| |{:^18}|{:^9.4f}|{:^9.4f}|{:^9.4f}|{:^9.4f}|"
          .format('filtered', mrr_filter.item() * 100, hits_value_filtered[0], hits_value_filtered[1],
                  hits_value_filtered[2], 'filtered', mrr_filter_r.item() * 100, hits_value_filtered_r[0],
                  hits_value_filtered_r[1], hits_value_filtered_r[2]))
    return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r


def run_experiment(args, history_len=None, dropout=None, n_bases=None, angle=None, sem_rate=None,
                   task_weight=None):
    # load configuration for grid search the best configuration
    if history_len:
        args.train_history_len = history_len
        args.test_history_len = history_len
    if dropout:
        args.dropout = dropout
    if n_bases:
        args.n_bases = n_bases
    if angle:
        args.angle = angle
    if sem_rate:
        args.sematic_rate = sem_rate
    if task_weight:
        args.task_weight = task_weight

    # load graph data
    print("loading graph data")
    data = utils.load_data(args.dataset)
    train_list, train_times = utils.split_by_time(data.train)
    valid_list, valid_times = utils.split_by_time(data.valid)
    test_list, test_times = utils.split_by_time(data.test)

    num_nodes = data.num_nodes
    num_rels = data.num_rels
    if args.dataset == "ICEWS14s":
        num_times = len(train_list) + len(valid_list) + len(test_list) + 1
    else:
        num_times = len(train_list) + len(valid_list) + len(test_list)
    time_interval = train_times[1] - train_times[0]
    print("num_times", num_times, "--------------", time_interval)
    history_val_time_nogt = valid_times[0]
    history_test_time_nogt = test_times[0]
    if args.multi_step:
        print("val only use global history before:", history_val_time_nogt)
        print("test only use global history before:", history_test_time_nogt)

    all_ans_list_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, False)
    all_ans_list_r_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, True)
    all_ans_list_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, False)
    all_ans_list_r_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, True)
    if args.add_static_graph:
        model_name = "{}-{}-{}-{}-dilate{}-his{}-dp{}-input_dp{}-hidden_dp{}-feature_dp{}-epoch{:04d}-weight{}" \
                     "-sen_rate{}-angle{}-discount{}-{}" \
            .format(args.model, args.dataset, args.encoder, args.decoder, args.dilate_len,
                    args.test_history_len, args.dropout, args.input_dropout, args.hidden_dropout, args.feat_dropout,
                    args.n_epochs, args.task_weight,args.sematic_rate, args.angle, args.discount,
                    args.history_method)
    else:
        model_name = "{}-{}-{}-{}-dilate{}-his{}-dp{}-input_dp{}-hidden_dp{}-feature_dp{}-epoch{:04d}-weight{}" \
                     "-sen_rate{}-{}" \
            .format(args.model, args.dataset, args.encoder, args.decoder, args.dilate_len,
                    args.test_history_len, args.dropout, args.input_dropout, args.hidden_dropout, args.feat_dropout,
                    args.n_epochs, args.task_weight, args.sematic_rate,args.history_method)
    model_state_file = '../models/' + args.dataset+'/' + model_name
    print("Sanity Check: stat name : {}".format(model_state_file))
    print("Sanity Check: Is cuda available ? {}".format(torch.cuda.is_available()))
    if os.path.exists(model_state_file):
        pass

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()

    if args.add_static_graph:
        static_triples = np.array(
            _read_triplets_as_list("../data/" + args.dataset + "/e-w-graph.txt", {}, {}, load_time=False))
        num_static_rels = len(np.unique(static_triples[:, 1]))
        num_words = len(np.unique(static_triples[:, 2]))
        static_triples[:, 2] = static_triples[:, 2] + num_nodes
        static_node_id = torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long().cuda(args.gpu) \
            if use_cuda else torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long()
        static_graph = build_sub_graph(len(static_node_id), num_static_rels, static_triples, use_cuda, args.gpu)
    else:
        num_static_rels, num_words, static_triples, static_graph = 0, 0, [], None
        static_graph = None

    model = None
    # create stat
    print("using model {}".format(args.model))
    if args.model == 'resegnn':
        model = RESEGNN(args.decoder,
                        args.encoder,
                        num_nodes,
                        num_rels,
                        num_static_rels,
                        num_words,
                        num_times,
                        time_interval,
                        args.n_hidden,
                        args.opn,
                        sequence_len=args.test_history_len,
                        num_bases=args.n_bases,
                        num_basis=args.n_basis,
                        dropout=args.dropout,
                        self_loop=args.self_loop,
                        skip_connect=args.skip_connect,
                        layer_norm=args.layer_norm,
                        input_dropout=args.input_dropout,
                        hidden_dropout=args.hidden_dropout,
                        feat_dropout=args.feat_dropout,
                        aggregation=args.aggregation,
                        use_static=args.add_static_graph,
                        entity_prediction=args.entity_prediction,
                        relation_prediction=args.relation_prediction,
                        use_cuda=use_cuda,
                        gpu=args.gpu,
                        sen_rate=args.sematic_rate,
                        angle=args.angle,
                        discount=args.discount,
                        weight=args.weight)
    else:
        raise NotImplementedError

    if use_cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda()

    # if args.add_static_graph:
    #     static_graph = build_sub_graph(len(static_node_id), num_static_rels, static_triples, use_cuda, args.gpu)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    if args.test and os.path.exists(model_state_file):
        mrr_raw_valid, mrr_filter_valid, mrr_raw_r_valid, mrr_filter_r_valid = test(model,
                                                                                    train_list + valid_list,
                                                                                    test_list,
                                                                                    num_rels,
                                                                                    num_nodes,
                                                                                    use_cuda,
                                                                                    all_ans_list_test,
                                                                                    all_ans_list_r_test,
                                                                                    model_state_file,
                                                                                    test_times,
                                                                                    history_test_time_nogt,
                                                                                    static_graph,
                                                                                    "test")
        return mrr_raw_valid, mrr_filter_valid, mrr_raw_r_valid, mrr_filter_r_valid
    elif args.test and not os.path.exists(model_state_file):
        print("--------------{} not exist, please train first----------------\n".format(model_state_file))
    else:
        print("----------------------------------------start training----------------------------------------")
        trained_n_epoch = 0
        best_mrr = 0
        best_valid_epoch = 0
        best_model_state_dict = model.state_dict()
        for n_epochs in range(args.n_epochs + 1, 0, -1):
            if args.add_static_graph:
                file_path = '../models/'+args.dataset + "/{}-{}-{}-{}-dilate{}-his{}-dp{}-input_dp{}-hidden_dp{}-feature_dp{}" \
                                           "-epoch{:04d}-weight{}-sen_rate{}-angle{}-discount{}-{}" \
                    .format(args.model, args.dataset, args.encoder, args.decoder, args.dilate_len,
                            args.test_history_len, args.dropout, args.input_dropout, args.hidden_dropout,
                            args.feat_dropout, n_epochs, args.task_weight, args.sematic_rate,
                            args.angle, args.discount,args.history_method)
            else:
                file_path = '../models/'+args.dataset + "/{}-{}-{}-{}-dilate{}-his{}-dp{}-input_dp{}-hidden_dp{}-feature_dp{}" \
                                           "-epoch{:04d}-weight{}-sen_rate{}-{}" \
                    .format(args.model, args.dataset, args.encoder, args.decoder, args.dilate_len,
                            args.test_history_len, args.dropout, args.input_dropout, args.hidden_dropout,
                            args.feat_dropout, n_epochs, args.task_weight, args.sematic_rate,args.history_method)
            if os.path.exists(file_path):
                if use_cuda:
                    checkpoint = torch.load(file_path, map_location=torch.device(args.gpu))
                else:
                    checkpoint = torch.load(file_path, map_location=torch.device('cpu'))
                best_model_state_dict = checkpoint['state_dict']
                model.load_state_dict(best_model_state_dict)
                best_valid_epoch = checkpoint['epoch']
                best_mrr = checkpoint['best_mrr']
                print('trained {} epoch found,using the model file to continue training'.format(best_valid_epoch))
                trained_n_epoch = best_valid_epoch
                break

        for epoch in range(trained_n_epoch + 1, args.n_epochs + 1):
            print("Epoch {:04d} training:".format(epoch))
            model.train()
            losses = []
            losses_e = []
            losses_r = []
            losses_static = []

            # idx = [_ for _ in range(len(train_list))]
            # random.shuffle(idx)

            train_data = SematicTrainDataset2(train_list, args.test_history_len, num_nodes, num_rels, train_times,
                                             args.dataset, args.history_method, use_cuda, args.gpu)
            train_dataloader = DataLoader(dataset=train_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,
                                          collate_fn=mycollate, prefetch_factor=10, pin_memory=True,persistent_workers=True)
            for input_list, start_history, output, tail_seq, rel_seq,tail_edge_seq, rel_edge_seq in tqdm(train_dataloader):
                for input_list_i, start_history_i, output_i, tail_seq_i, rel_seq_i,tail_edge_seq_i, rel_edge_seq_i \
                        in zip(input_list, start_history,output, tail_seq, rel_seq,tail_edge_seq, rel_edge_seq):
                    if len(input_list_i) == 0:
                        continue
                    history_glist = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu) for snap in input_list_i]
                    # if use_cuda:
                    #     tail_seq_i = tail_seq_i.cuda()
                    #     rel_seq_i = rel_seq_i.cuda()

                    tail_seq_i = torch.sparse.FloatTensor(*tail_seq_i)
                    rel_seq_i = torch.sparse.FloatTensor(*rel_seq_i)
                    if 'edge' in args.history_method:
                        tail_edge_seq_i = torch.sparse.FloatTensor(*tail_edge_seq_i)
                        rel_edge_seq_i = torch.sparse.FloatTensor(*rel_edge_seq_i)
                    if use_cuda:
                        tail_seq_i = tail_seq_i.cuda()
                        rel_seq_i = rel_seq_i.cuda()
                        if 'edge' in args.history_method:
                            tail_edge_seq_i = tail_edge_seq_i.cuda()
                            rel_edge_seq_i = rel_edge_seq_i.cuda()
                    tail_seq_i = tail_seq_i.to_dense()
                    rel_seq_i = rel_seq_i.to_dense()
                    if 'edge' in args.history_method:
                        tail_edge_seq_i = tail_edge_seq_i.to_dense()
                        rel_edge_seq_i = rel_edge_seq_i.to_dense()

                    loss_e, loss_r, loss_static = model.get_loss(history_glist, start_history_i, output_i[0],
                                                                 static_graph,tail_seq_i, rel_seq_i,tail_edge_seq_i,rel_edge_seq_i)
                    loss = args.task_weight * loss_e + (1 - args.task_weight) * loss_r + loss_static

                    losses.append(loss.item())
                    losses_e.append(loss_e.item())
                    losses_r.append(loss_r.item())
                    losses_static.append(loss_static.item())

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
                    optimizer.step()
                    optimizer.zero_grad()

            print(
                "Epoch {:04d} result: Ave Loss: {:.4f} | entity-relation:{:.4f}-{:.4f} Best MRR {:.4f} | Model {} "
                .format(epoch, np.mean(losses), np.mean(losses_e), np.mean(losses_r), best_mrr, model_name))

            # validation
            if epoch % args.evaluate_every == 0:
                mrr_raw_valid, mrr_filter_valid, mrr_raw_r_valid, mrr_filter_r_valid = \
                    test(model, train_list, valid_list, num_rels, num_nodes, use_cuda, all_ans_list_valid,
                         all_ans_list_r_valid, model_state_file, valid_times, history_val_time_nogt, static_graph,
                         mode="valid")

                if not args.relation_evaluation:  # entity prediction evalution
                    if mrr_filter_valid < best_mrr:
                        if epoch >= args.n_epochs or args.end_by == 'local_best':
                            break
                    else:
                        best_mrr = mrr_filter_valid
                        best_valid_epoch = epoch
                        best_model_state_dict = model.state_dict()
                        # torch.save({'state_dict': model.state_dict(), 'epoch': best_valid_epoch,'best_mrr': best_mrr},
                        #            model_state_file)
                else:
                    if mrr_filter_r_valid < best_mrr:
                        if epoch >= args.n_epochs or args.end_by == 'local_best':
                            break
                    else:
                        best_mrr = mrr_filter_r_valid
                        best_valid_epoch = epoch
                        best_model_state_dict = model.state_dict()
                        # torch.save({'state_dict': model.state_dict(), 'epoch': best_valid_epoch,'best_mrr': best_mrr},
                        #            model_state_file)
            print("-----------end of epoch {}---------".format(epoch))
        if not os.path.exists('../models/{}/'.format(args.dataset)):
            os.makedirs('../models/{}/'.format(args.dataset))
        torch.save({
            'state_dict': best_model_state_dict,
            'epoch': best_valid_epoch,
            'best_mrr': best_mrr
        },
            model_state_file)
        # test
        mrr_raw_test, mrr_filter_test, mrr_raw_r_test, mrr_filter_r_test = \
            test(model, train_list + valid_list, test_list, num_rels, num_nodes, use_cuda, all_ans_list_test,
                 all_ans_list_r_test, model_state_file, test_times, history_test_time_nogt, static_graph, mode="test")
        return mrr_raw_test, mrr_filter_test, mrr_raw_r_test, mrr_filter_r_test, best_valid_epoch


def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    """
    将scipy的sparse matrix转换成torch的sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='REGCN')

    parser.add_argument("--gpu", default=-1, type=int, help='gpu for CUDA training')
    parser.add_argument("--batch-size", type=int, default=10, help="batch-size")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="dataset to use")
    parser.add_argument("--test", action='store_true', default=False, help="load stat from dir and directly test")
    parser.add_argument("--add-static-graph", action='store_true', default=False, help="use the info of static graph")
    parser.add_argument("--multi-step", action='store_true', default=False,
                        help="do multi-steps inference without ground truth")
    parser.add_argument("--topk", type=int, default=10,
                        help="choose top k entities as results when do multi-steps without ground truth")
    parser.add_argument("--add-rel-word", action='store_true', default=False, help="use words in relaitons")
    parser.add_argument("--relation-evaluation", action='store_true', default=False,
                        help="save model accordding to the relation evalution")

    # configuration for encoder RGCN stat
    parser.add_argument("--weight", type=float, default=0.5,
                        help="weight of static constraint")
    parser.add_argument("--task-weight", type=float, default=0,
                        help="weight of entity prediction task")
    parser.add_argument("--discount", type=float, default=1,
                        help="discount of weight of static constraint")
    parser.add_argument("--angle", type=int, default=10,
                        help="evolution speed")

    parser.add_argument("--encoder", type=str, default="uvrgcn",
                        help="method of encoder")
    parser.add_argument("--aggregation", type=str, default="none",
                        help="method of aggregation")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--skip-connect", action='store_true', default=False,
                        help="whether to use skip connect in a RGCN Unit")
    parser.add_argument("--n-hidden", type=int, default=200,
                        help="number of hidden units")
    parser.add_argument("--opn", type=str, default="sub",
                        help="opn of compgcn")

    parser.add_argument("--n-bases", type=int, default=100,
                        help="number of weight blocks for each relation")
    parser.add_argument("--n-basis", type=int, default=100,
                        help="number of basis vector for compgcn")
    parser.add_argument("--self-loop", action='store_true', default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--layer-norm", action='store_true', default=False,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--relation-prediction", action='store_true', default=False,
                        help="add relation prediction loss")
    parser.add_argument("--entity-prediction", action='store_true', default=False,
                        help="add entity prediction loss")
    parser.add_argument("--split_by_relation", action='store_true', default=False,
                        help="do relation prediction")

    # configuration for stat training
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="number of minimum training epochs on each time step")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")

    # configuration for evaluating
    parser.add_argument("--evaluate-every", type=int, default=50,
                        help="perform evaluation every n epochs")

    # configuration for decoder
    parser.add_argument("--decoder", type=str, default="convtranse", help="method of decoder")
    parser.add_argument("--input-dropout", type=float, default=0.2, help="input dropout for decoder ")
    parser.add_argument("--hidden-dropout", type=float, default=0.2, help="hidden dropout for decoder")
    parser.add_argument("--feat-dropout", type=float, default=0.2, help="feat dropout for decoder")

    # configuration for sequences stat
    parser.add_argument("--train-history-len", type=int, default=10, help="history length")
    parser.add_argument("--test-history-len", type=int, default=10, help="history length for test")
    parser.add_argument("--dilate-len", type=int, default=1, help="dilate history graph")

    # configuration for optimal parameters
    parser.add_argument("--grid-search", action='store_true', default=False,
                        help="perform grid search for best configuration")
    parser.add_argument("-tune", "--tune", type=str,
                        default="history_len,dropout,n_bases,angle,sem_rate,task_weight",
                        help="stat to use")
    parser.add_argument("--num-k", type=int, default=500, help="number of triples generated")

    # model
    parser.add_argument("--model", default='segnn', help="model to use")

    # configuration for segnn
    parser.add_argument("--num-ent-drop", default=0.3, type=float, help="segnn entity dropout")
    parser.add_argument("--num-rel-drop", default=0.1, type=float, help="segnn relation dropout")
    parser.add_argument("--sematic-rate", default=0, type=float, help="segnn sematic rate")
    parser.add_argument("--history-method", default="time_decay", type=str, help="sematic method")

    # configuration for debug
    parser.add_argument("--end-by", default="local_best", type=str, help="which epoch to end training")
    parser.add_argument("--num-workers", default=8, type=int, help="dataloader num_workers")
    args = parser.parse_args()
    print(args)
    # torch.multiprocessing.set_start_method('spawn')
    if args.grid_search:
        if not os.path.exists('./grid_search/{}/'.format(args.dataset)):
            os.makedirs('./grid_search/{}/'.format(args.dataset))
        out_log = './grid_search/{}/{}.{}.{}.gs'.format(args.dataset, args.dataset, args.model,
                                          time.strftime('%m-%d-%H-%M-%S', time.localtime()))
        o_f = open(out_log, 'w')
        print("** Grid Search **")
        o_f.write("** Grid Search **\n")
        hyperparameters = args.tune.split(',')

        if args.tune == '' or len(hyperparameters) < 1:
            print("No hyperparameter specified.")
            sys.exit(0)
        if args.dataset == "ICEWS14":
            hp_range = hp_range_ICEWS14
        elif args.dataset == "WIKI":
            hp_range = hp_range_WIKI
        elif args.dataset == "YAGO":
            hp_range = hp_range_YAGO
        elif args.dataset == "ICEWS18":
            hp_range = hp_range_ICEWS18
        elif args.dataset == "ICEWS05-15":
            hp_range = hp_range_ICEWS05_15
        elif args.dataset == "GDELT":
            hp_range = hp_range_GDELT
        else:
            hp_range = hp_range_ICEWS14
        grid = hp_range[hyperparameters[0]]
        for hp in hyperparameters[1:]:
            grid = itertools.product(grid, hp_range[hp])
        hits_at_1s = {}
        hits_at_10s = {}
        mrrs = {}
        grid = list(grid)
        o_f.write(str(args) + '\n')
        o_f.write('tune:' + args.tune + '\n')
        print('* {} hyperparameter combinations to try'.format(len(grid)))
        o_f.write('* {} hyperparameter combinations to try\n'.format(len(grid)))
        o_f.close()

        for i, grid_entry in enumerate(list(grid)):
            o_f = open(out_log, 'a')
            if not (type(grid_entry) is list or type(grid_entry) is list):
                grid_entry = [grid_entry]
            grid_entry = utils.flatten(grid_entry)
            print('* Hyperparameter Set {}:'.format(i))
            o_f.write('* Hyperparameter Set {}:\n'.format(i))
            signature = ''
            print(grid_entry)
            o_f.write("\t".join([str(_) for _ in grid_entry]) + "\n")
            mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r, best_epoch = run_experiment(args, history_len=grid_entry[0],
                                                                                      dropout=grid_entry[1],
                                                                                      n_bases=grid_entry[2],
                                                                                      angle=grid_entry[3],
                                                                                      sem_rate=grid_entry[4],
                                                                                      task_weight=grid_entry[5])
            o_f.write("best epoch|total epoch:\t{:02} |{:02} \n".format(best_epoch, args.n_epochs))
            print("best filtered|raw entity MRR:\t{:.2f}%|{:.2f}%".format(mrr_filter * 100, mrr_raw * 100))
            o_f.write("best filtered|raw entity MRR:\t{:.2f}%|{:.2f}%\n".format(mrr_filter * 100, mrr_raw * 100))
            print("best filtered|raw relation MRR:\t{:.2f}%|{:.2f}%".format(mrr_filter_r * 100, mrr_raw_r * 100))
            o_f.write("best filtered|raw relation MRR:\t{:.2f}%|{:.2f}%\n".format(mrr_filter_r * 100, mrr_raw_r * 100))
            o_f.close()
    # single run
    else:
        run_experiment(args)
    sys.exit()
