import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from rgcn.layers import UnionRGCNLayer, RGCNBlockLayer
from src.gcn import BaseRGCN
from rgcn.layers import RGCNBlockLayer
from src.SEGNN import SEGNN
from src.decoder import ConvTransEEdgeParted, ConvTransREdgeParted, ConvTransE, ConvTransR, ConvTransEEdge, \
    ConvTransREdge
# from tools.get_atten import *
# from tools.get_para import *

class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                                  activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc,
                                  rel_emb=self.rel_emb)
        else:
            raise NotImplementedError

    def forward(self, g, init_ent_emb, init_rel_emb):
        # 消息传递
        if self.encoder_name in "uvrgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r[i])
            return g.ndata.pop('h')
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')


class RESEGNN(nn.Module):
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, num_static_rels, num_words, num_times,
                 time_interval, h_dim, opn, sequence_len, num_bases=-1, num_basis=-1,sen_rate=0.3,
                dropout=0, self_loop=False,
                 skip_connect=False, layer_norm=False, input_dropout=0,
                 hidden_dropout=0, feat_dropout=0, aggregation='cat', weight=1, discount=0, angle=0, use_static=False,
                 entity_prediction=False, relation_prediction=False, use_cuda=False, gpu=0, analysis=False):
        super(RESEGNN, self).__init__()
        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.num_words = num_words
        self.num_static_rels = num_static_rels
        self.num_times = num_times
        self.time_interval = time_interval
        self.total_time = num_times * time_interval
        self.opn = opn
        self.sen_rate = sen_rate
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.aggregation = aggregation
        # self.relation_evolve = False
        self.discount = discount
        self.angle = angle
        self.use_static = use_static
        self.relation_prediction = relation_prediction
        self.entity_prediction = entity_prediction
        self.use_cuda = use_cuda
        self.gpu = gpu
        self.weight = weight

        self.w1 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w1)

        self.w2 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w2)

        self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.emb_rel)
        self.h_r = None

        self.dynamic_emb = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float()
        torch.nn.init.normal_(self.dynamic_emb)
        self.h_e = None

        self.loss_r = torch.nn.CrossEntropyLoss()
        # self.loss_e = torch.nn.CrossEntropyLoss()

        self.rgcn = RGCNCell(num_ents, h_dim, h_dim, num_rels * 2, num_bases, num_basis, 2, dropout,
                             self_loop, skip_connect, encoder_name, self.opn, self.emb_rel, use_cuda, analysis)

        if self.use_static:
            self.words_emb = torch.nn.Parameter(torch.Tensor(self.num_words, h_dim), requires_grad=True).float()
            torch.nn.init.xavier_normal_(self.words_emb)
            self.static_rgcn_layer = RGCNBlockLayer(self.h_dim, self.h_dim, self.num_static_rels * 2, num_bases,
                                                    activation=F.rrelu, dropout=dropout, self_loop=False,
                                                    skip_connect=False)
            self.static_loss = torch.nn.MSELoss()

        self.segnn = SEGNN(num_ents, num_rels, h_dim=h_dim, device=gpu)

        self.e_time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))
        nn.init.xavier_uniform_(self.e_time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.e_time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.e_time_gate_bias)

        self.r_time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))
        nn.init.xavier_uniform_(self.r_time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.r_time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.r_time_gate_bias)

        # GRU cell for relation evolving
        self.relation_cell_1 = nn.GRUCell(self.h_dim, self.h_dim)
        self.entity_cell_1 = nn.GRUCell(self.h_dim, self.h_dim)

        # decoder
        if decoder_name == "convtranse":
            self.decoder_ob = ConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.decoder_ob_sematic = ConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.decoder_rel = ConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.decoder_rel_sematic = ConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
        elif decoder_name == "convtranse_edge":
            # consider edge and node together
            self.decoder_ob = ConvTransEEdge(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.decoder_ob_sematic = ConvTransEEdge(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.decoder_rel = ConvTransREdge(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.decoder_rel_sematic = ConvTransREdge(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
        elif decoder_name == "convtranse_edge_parted":
            # consider edge and node separately
            self.decoder_ob = ConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.decoder_ob_sematic = ConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.decoder_ob_edge_sematic = ConvTransEEdgeParted(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.decoder_rel = ConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.decoder_rel_sematic = ConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.decoder_rel_edge_sematic = ConvTransREdgeParted(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
        else:
            raise NotImplementedError

    def forward(self, g_list, start_history, static_graph):
        if self.use_static:
            static_graph = static_graph.to(self.gpu) if self.use_cuda else static_graph
            static_graph.ndata['h'] = torch.cat((self.dynamic_emb, self.words_emb), dim=0)  # 演化得到的表示，和wordemb满足静态图约束
            self.static_rgcn_layer(static_graph, [])
            static_emb = static_graph.ndata.pop('h')[:self.num_ents, :]
            static_emb = F.normalize(static_emb) if self.layer_norm else static_emb
            self.h_e = static_emb
        else:
            self.h_e = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
            static_emb = None

        history_ent_embedding = []

        for i, g in enumerate(g_list):
            if self.use_cuda:
                g = g.to(self.gpu)
                temp_e = self.h_e[g.r_to_e.type(torch.long)]
            else:
                temp_e = self.h_e[g.r_to_e]
            x_input = torch.zeros(self.num_rels * 2, self.h_dim).float().cuda() if self.use_cuda else torch.zeros(
                self.num_rels * 2, self.h_dim).float()
            for span, r_idx in zip(g.r_len, g.uniq_r):
                x = temp_e[span[0]:span[1], :]
                x_mean = torch.mean(x, dim=0, keepdim=True)
                x_input[r_idx] = x_mean
            if i == 0:
                current_r = self.relation_cell_1(x_input, self.emb_rel)  # 第1层输入
            else:
                current_r = self.relation_cell_1(x_input, self.h_r)  # 第2层输出==下一时刻第一层输入

            current_h, current_r = self.segnn.forward(g, self.h_e, current_r)
            current_h = F.normalize(current_h) if self.layer_norm else current_h
            current_r = F.normalize(current_r) if self.layer_norm else current_r

            self.h_e = self.entity_cell_1(current_h, self.h_e)
            self.h_e = F.normalize(self.h_e) if self.layer_norm else self.h_e

            e_time_weight = torch.sigmoid(torch.mm(self.h_e, self.e_time_gate_weight) + self.e_time_gate_bias)
            self.h_e = e_time_weight * current_h + (1 - e_time_weight) * self.h_e
            if i == 0:
                r_time_weight = torch.sigmoid(torch.mm(current_r, self.r_time_gate_weight) + self.r_time_gate_bias)
                self.h_r = r_time_weight * current_r + (1 - r_time_weight) * self.emb_rel
            else:
                r_time_weight = torch.sigmoid(torch.mm(current_r, self.r_time_gate_weight) + self.r_time_gate_bias)
                self.h_r = r_time_weight * current_r + (1 - r_time_weight) * self.h_r
            history_ent_embedding.append(self.h_e)
        return history_ent_embedding, static_emb, self.h_r

    def predict(self, test_graph, start_history, num_rels, static_graph, test_triplets, ent_sematic, rel_sematic,
                ent_edge_sematic=None, rel_edge_sematic=None):
        with torch.no_grad():
            inverse_test_triplets = test_triplets[:, [2, 1, 0, 3]]
            inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels  # 将逆关系换成逆关系的id
            all_triples = torch.cat((test_triplets, inverse_test_triplets))
            # 历史每一个时刻输出,_,关系嵌入
            evolve_ent_embeddings, _, evolve_rel_embedding = self.forward(test_graph, start_history, static_graph)
            evolve_ent_embedding = evolve_ent_embeddings[-1]
            evolve_ent_embedding = F.normalize(evolve_ent_embedding) if self.layer_norm else evolve_ent_embedding

            # evolve_rel_embedding = attention_rel_embedding
            evolve_rel_embedding = F.normalize(evolve_rel_embedding) if self.layer_norm else evolve_rel_embedding
            # convTransE
            score_ent = self.decoder_ob.forward(evolve_ent_embedding, evolve_rel_embedding, all_triples, mode="test")
            score_ent = F.log_softmax(score_ent, dim=1)
            score_ent_sematic = self.decoder_ob_sematic \
                .forward(evolve_ent_embedding, evolve_rel_embedding, all_triples, partial_embedding=ent_sematic,
                         partial_embedding_edge=ent_edge_sematic, mode="test")
            score_ent_sematic = F.log_softmax(score_ent_sematic, dim=1)
            # score_ent = torch.log(score_ent)

            score_rel = self.decoder_rel.forward(evolve_ent_embedding, evolve_rel_embedding, all_triples, mode="test")
            score_rel = F.log_softmax(score_rel, dim=1)
            score_rel_sematic = self.decoder_rel_sematic \
                .forward(evolve_ent_embedding, evolve_rel_embedding, all_triples, partial_embedding=rel_sematic,
                         partial_embedding_edge=rel_edge_sematic, mode="test")
            score_rel_sematic = F.log_softmax(score_rel_sematic, dim=1)

            if self.decoder_name == "convtranse_edge_parted":
                score_ent_edge_sematic = self.decoder_ob_edge_sematic \
                    .forward(evolve_ent_embedding, evolve_rel_embedding, all_triples, partial_embedding=ent_sematic,
                             partial_embedding_edge=ent_edge_sematic, mode="test")
                if score_ent_edge_sematic is not None:
                    score_ent_edge_sematic = F.log_softmax(score_ent_edge_sematic, dim=1)
                    score_ent = self.sen_rate * (score_ent_sematic + score_ent_edge_sematic) + (1 - self.sen_rate) * score_ent
                else:
                    score_ent = self.sen_rate * score_ent_sematic + (1 - self.sen_rate) * score_ent

                score_rel_edge_sematic = self.decoder_rel_edge_sematic \
                    .forward(evolve_ent_embedding, evolve_rel_embedding, all_triples, partial_embedding=rel_sematic,
                             partial_embedding_edge=rel_edge_sematic, mode="test")
                if score_rel_edge_sematic is not None:
                    score_rel_edge_sematic = F.log_softmax(score_rel_edge_sematic, dim=1)
                    score_rel = self.sen_rate * (score_rel_sematic + score_rel_edge_sematic) + (1 - self.sen_rate) * score_rel
                else:
                    score_rel = self.sen_rate * score_rel_sematic + (1 - self.sen_rate) * score_rel
            else:
                score_ent = self.sen_rate * score_ent_sematic + (1 - self.sen_rate) * score_ent
                score_rel = self.sen_rate * score_rel_sematic + (1 - self.sen_rate) * score_rel
            # score_rel = torch.log(score_rel)
            return all_triples, score_ent, score_rel

    def get_loss(self, glist, start_history, triples, static_graph, ent_sematic, rel_sematic, ent_edge_sematic=None,
                 rel_edge_sematic=None):
        loss_ent = torch.zeros(1).cuda().to(self.gpu) if self.use_cuda else torch.zeros(1)
        loss_rel = torch.zeros(1).cuda().to(self.gpu) if self.use_cuda else torch.zeros(1)
        loss_static = torch.zeros(1).cuda().to(self.gpu) if self.use_cuda else torch.zeros(1)

        # 将预期的输出加入逆关系
        inverse_triples = triples[:, [2, 1, 0, 3]]
        # 逆关系等于原来关系编码+关系数
        inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
        # 逆关系三元组加入预期输出中
        all_triples = torch.cat([triples, inverse_triples])
        if self.use_cuda:
            all_triples = all_triples.to(self.gpu)

        evolve_ent_embeddings, static_emb, evolve_rel_embedding = self.forward(glist, start_history, static_graph)
        evolve_ent_embedding = evolve_ent_embeddings[-1]
        evolve_ent_embedding = F.normalize(evolve_ent_embedding) if self.layer_norm else evolve_ent_embedding

        evolve_rel_embedding = F.normalize(evolve_rel_embedding) if self.layer_norm else evolve_rel_embedding

        if self.entity_prediction:
            scores_ob = self.decoder_ob.forward(evolve_ent_embedding, evolve_rel_embedding, all_triples).view(-1,
                                                                                                              self.num_ents)

            scores_ob = F.log_softmax(scores_ob, dim=1)
            scores_ob_sematic = self.decoder_ob_sematic.forward(evolve_ent_embedding, evolve_rel_embedding, all_triples,
                                                                partial_embedding=ent_sematic,
                                                                partial_embedding_edge=ent_edge_sematic)
            scores_ob_sematic = F.log_softmax(scores_ob_sematic, dim=1)

            if self.decoder_name == "convtranse_edge_parted":
                scores_ob_edge_sematic = self.decoder_ob_edge_sematic.forward(evolve_ent_embedding,
                                                                              evolve_rel_embedding, all_triples,
                                                                              partial_embedding=ent_sematic,
                                                                              partial_embedding_edge=ent_edge_sematic)
                if scores_ob_edge_sematic is not None:
                    scores_ob_edge_sematic = F.log_softmax(scores_ob_edge_sematic, dim=1)
                    score_en_all = self.sen_rate * (scores_ob_sematic + scores_ob_edge_sematic) + (1 - self.sen_rate) * scores_ob
                else:
                    score_en_all = self.sen_rate * scores_ob_sematic + (1 - self.sen_rate) * scores_ob
            else:
                score_en_all = self.sen_rate * scores_ob_sematic + (1 - self.sen_rate) * scores_ob
            loss_ent += F.nll_loss(score_en_all, all_triples[:, 2])

        if self.relation_prediction:
            score_rel = self.decoder_rel.forward(evolve_ent_embedding, evolve_rel_embedding, all_triples).view(-1,2 * self.num_rels)
            score_rel = F.log_softmax(score_rel, dim=1)
            score_rel_sematic = self.decoder_rel_sematic.forward(evolve_ent_embedding, evolve_rel_embedding,
                                                                 all_triples, partial_embedding=rel_sematic,
                                                                 partial_embedding_edge=rel_edge_sematic)
            score_rel_sematic = F.log_softmax(score_rel_sematic, dim=1)
            if self.decoder_name == "convtranse_edge_parted":
                score_rel_edge_sematic = self.decoder_rel_edge_sematic.forward(evolve_ent_embedding,
                                                                               evolve_rel_embedding,
                                                                               all_triples,
                                                                               partial_embedding=rel_sematic,
                                                                               partial_embedding_edge=rel_edge_sematic)
                if score_rel_edge_sematic is not None:
                    score_rel_edge_sematic = F.log_softmax(score_rel_edge_sematic, dim=1)
                    score_rel_all = self.sen_rate * (score_rel_sematic + score_rel_edge_sematic) + (1 - self.sen_rate) * score_rel
                else:
                    score_rel_all = self.sen_rate * score_rel_sematic + (1 - self.sen_rate) * score_rel
            else:
                score_rel_all = self.sen_rate * score_rel_sematic + (1 - self.sen_rate) * score_rel
            loss_rel += F.nll_loss(score_rel_all, all_triples[:, 1])

        if self.use_static:
            if self.discount == 1:
                for time_step, evolve_emb in enumerate(evolve_ent_embeddings):
                    angle = 90 // len(evolve_ent_embeddings)
                    # step = (self.angle * math.pi / 180) * (time_step + 1)
                    step = (self.angle * math.pi / 180) * (time_step + 1)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
            elif self.discount == 0:
                for time_step, evolve_emb in enumerate(evolve_ent_embeddings):
                    step = (self.angle * math.pi / 180)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))

        return loss_ent, loss_rel, loss_static
