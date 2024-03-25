import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from torch.nn import Parameter
from torch.nn.init import xavier_normal_


def get_param(*shape):
    param = Parameter(torch.zeros(shape))
    xavier_normal_(param)
    return param


class SEGNN(nn.Module):
    def __init__(self, num_ent, num_rel, h_dim=450, device=0, bn=False, ent_drop=0.3, rel_drop=0.1):
        super().__init__()
        self.device = device
        self.n_ent = num_ent
        self.n_rel = num_rel
        # tes
        self.rel_w = get_param(h_dim, h_dim)

        self.edge_layers = EdgeLayer(h_dim, num_ent, num_rel, device=device, bn=bn)
        self.node_layers = NodeLayer(h_dim, num_ent, num_rel, device=device, bn=bn)
        self.comp_layers = CompLayer(h_dim, num_ent, num_rel, device=device, bn=bn)
        # loss
        self.bce = nn.BCELoss()

        self.ent_drop = nn.Dropout(ent_drop)
        self.rel_drop = nn.Dropout(rel_drop)
        self.act = nn.Tanh()

    def forward(self, kg, ent_enb, rel_emb):
        """
        matching computation between query (h, r) and answer t.
        :param h_id: head entity id, (bs, )
        :param r_id: relation id, (bs, )
        :param kg: aggregation graph
        :return: matching score, (bs, n_ent)
        """
        assert rel_emb.shape[0] == self.n_rel*2
        # aggregate embedding
        ent_emb,rel_emb = self.aggragate_emb(kg, ent_enb, rel_emb)
        return ent_emb,rel_emb

    def loss(self, score, label):
        # (bs, n_ent)
        loss = self.bce(score, label)

        return loss

    def aggragate_emb(self, kg, init_ent_enb, init_rel_emb):
        """
        aggregate embedding.
        :param kg:
        :return:
        """
        assert init_rel_emb.shape[0] == self.n_rel*2

        ent_emb, rel_emb = self.ent_drop(init_ent_enb), self.rel_drop(init_rel_emb)
        edge_ent_emb = self.edge_layers(kg, ent_emb, rel_emb)
        node_ent_emb = self.node_layers(kg, ent_emb)
        comp_ent_emb = self.comp_layers(kg, ent_emb, rel_emb)
        ent_emb = ent_emb + edge_ent_emb + node_ent_emb + comp_ent_emb
        rel_emb = rel_emb.mm(self.rel_w)
        return ent_emb,rel_emb


class CompLayer(nn.Module):
    def __init__(self, h_dim, num_ent, num_rel, comp_op='mul', device=0, bn=False):
        super().__init__()
        self.device = device
        self.n_ent = num_ent
        self.n_rel = num_rel
        self.comp_op = comp_op
        assert self.comp_op in ['add', 'mul']

        self.neigh_w = get_param(h_dim, h_dim)
        self.act = nn.Tanh()
        if bn:
            self.bn = torch.nn.BatchNorm1d(h_dim)
        else:
            self.bn = None

    def forward(self, kg, ent_emb, rel_emb):
        assert kg.number_of_nodes() == ent_emb.shape[0]
        assert rel_emb.shape[0] == self.n_rel*2

        with kg.local_scope():
            node_id = kg.ndata['id'].squeeze()
            kg.ndata['h'] = ent_emb[node_id]
            rel_id = kg.edata['type'].squeeze()
            kg.edata['h'] = rel_emb[rel_id]
            if self.comp_op == 'add':
                kg.apply_edges(fn.u_add_e('h', 'h', 'comp_emb'))
            elif self.comp_op == 'mul':
                kg.apply_edges(fn.u_mul_e('h', 'h', 'comp_emb'))
            else:
                raise NotImplementedError

            # attention
            kg.apply_edges(fn.e_dot_v('comp_emb', 'h', 'norm'))  # (n_edge, 1)
            kg.edata['norm'] = dgl.ops.edge_softmax(kg, kg.edata['norm'])
            # agg
            kg.edata['comp_emb'] = kg.edata['comp_emb'] * kg.edata['norm']
            kg.update_all(fn.copy_e('comp_emb', 'm'), fn.sum('m', 'neigh'))
            neigh_ent_emb = kg.ndata['neigh']
            neigh_ent_emb = neigh_ent_emb.mm(self.neigh_w)

            if callable(self.bn):
                neigh_ent_emb = self.bn(neigh_ent_emb)

            neigh_ent_emb = self.act(neigh_ent_emb)

        return neigh_ent_emb


class NodeLayer(nn.Module):
    def __init__(self, h_dim, num_ent, num_rel, device=0, bn=False):
        super().__init__()
        self.device = device
        self.n_ent = num_ent
        self.n_rel = num_rel

        self.neigh_w = get_param(h_dim, h_dim)
        self.act = nn.Tanh()
        if bn:
            self.bn = torch.nn.BatchNorm1d(h_dim)
        else:
            self.bn = None

    def forward(self, kg, ent_emb):
        assert kg.number_of_nodes() == ent_emb.shape[0]

        with kg.local_scope():
            node_id = kg.ndata['id'].squeeze()
            kg.ndata['h'] = ent_emb[node_id]

            # attention
            kg.apply_edges(fn.u_dot_v('h', 'h', 'norm'))  # (n_edge, 1)
            kg.edata['norm'] = dgl.ops.edge_softmax(kg, kg.edata['norm'])

            # agg
            kg.update_all(fn.u_mul_e('h', 'norm', 'm'), fn.sum('m', 'neigh'))
            neigh_ent_emb = kg.ndata['neigh']

            neigh_ent_emb = neigh_ent_emb.mm(self.neigh_w)

            if callable(self.bn):
                neigh_ent_emb = self.bn(neigh_ent_emb)

            neigh_ent_emb = self.act(neigh_ent_emb)

        return neigh_ent_emb


class EdgeLayer(nn.Module):
    def __init__(self, h_dim, num_ent, num_rel, device=0, bn=False):
        super().__init__()
        self.device = device
        self.n_ent = num_ent
        self.n_rel = num_rel

        self.neigh_w = get_param(h_dim, h_dim)
        self.act = nn.Tanh()
        if bn:
            self.bn = torch.nn.BatchNorm1d(h_dim)
        else:
            self.bn = None

    def forward(self, kg, ent_emb, rel_emb):
        assert kg.number_of_nodes() == ent_emb.shape[0]
        assert rel_emb.shape[0] == self.n_rel*2

        with kg.local_scope():
            node_id = kg.ndata['id'].squeeze()
            kg.ndata['h'] = ent_emb[node_id]
            rel_id = kg.edata['type'].squeeze()
            kg.edata['h'] = rel_emb[rel_id]
            # attention
            kg.apply_edges(fn.e_dot_v('h', 'h', 'norm'))  # (n_edge, 1)
            kg.edata['norm'] = dgl.ops.edge_softmax(kg, kg.edata['norm'])
            # agg
            kg.edata['h'] = kg.edata['h'] * kg.edata['norm']
            kg.update_all(fn.copy_e('h', 'm'), fn.sum('m', 'neigh'))
            neigh_ent_emb = kg.ndata['neigh']
            neigh_ent_emb = neigh_ent_emb.mm(self.neigh_w)

            if callable(self.bn):
                neigh_ent_emb = self.bn(neigh_ent_emb)

            neigh_ent_emb = self.act(neigh_ent_emb)

        return neigh_ent_emb
