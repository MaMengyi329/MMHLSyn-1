#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# @Author: Scz 
# @Time:  2021/12/10 10:46
import torch
import torch.nn as nn
from torch_geometric.nn import HypergraphConv, GATConv, global_max_pool, global_mean_pool, GINConv, GCNConv
import sys
from Model.MMHLSyn.trans import MultiHeadAttention, FeedForward, CrossAttention
sys.path.append('..')
from utils import reset


drug_num = 87
cline_num = 55




class Interactive(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Interactive, self).__init__()
        self.rnn = nn.GRU(100, 1* 100 // 2, bidirectional=True)
        self.attn1 = nn.Linear(100, 100, bias=False)
        self.conv1 = HypergraphConv(in_channels, 512)
        self.batch1 = nn.BatchNorm1d(512)
        self.conv2 = HypergraphConv(512, out_channels)
        self.drop_out = nn.Dropout(0.3)
        self.act = nn.LeakyReLU(0.2)
        self.li=nn.Linear(200,100)
        self.MH = MultiHeadAttention(4, 100, 0.5, 0.5, 1e-5)
        self.FW = FeedForward(100, 256, 0.5, 'sigmoid', 1e-5)



    def forward(self, x, edge):
        x= self.MH(x, None)
        x= self.MH(x,None)

        x= self.act(self.conv1(x, edge))
        x= self.batch1(x)
        x = self.drop_out(x)
        x= self.act(self.conv2(x, edge))
        return x


class Atom_Gloal(nn.Module):
    def __init__(self, dim_drug, dim_cellline, output, use_GMP=True):
        super(Atom_Gloal, self).__init__()
        # -------drug_layer
        self.use_GMP = use_GMP
        self.conv1 = GCNConv(dim_drug, 128)
        self.batch_conv1 = nn.BatchNorm1d(128)
        self.conv2 = GCNConv(128,output )
        self.batch_conv2 = nn.BatchNorm1d(output)  # todo 新增一个batch norm
        self.fc_cell1 = nn.Linear(dim_cellline, 128)
        self.batch_cell1 = nn.BatchNorm1d(128)
        self.fc_cell2 = nn.Linear(128, output)
        self.drop_out = nn.Dropout(0.3)
        self.act = nn.ReLU()

    def forward(self, drug_feature, drug_adj, ibatch, gexpr_data):
        # -----drug_train
        x_drug = self.conv1(drug_feature, drug_adj)
        x_drug = self.batch_conv1(self.act(x_drug))
        x_drug = self.drop_out(x_drug)
        x_drug = self.conv2(x_drug, drug_adj)
        x_drug = self.batch_conv2(self.act(x_drug))
        if self.use_GMP:
            x_drug = global_max_pool(x_drug, ibatch)
        else:
            x_drug = global_mean_pool(x_drug, ibatch)

        # ----cellline_train
        # x_cellline=self.cell_conv(gexpr_data)
        # x_cellline=self.fc_cell1(x_cellline)
        x_cellline = torch.tanh(self.fc_cell1(gexpr_data))
        x_cellline = self.batch_cell1(x_cellline)
        x_cellline = self.drop_out(x_cellline)
        x_cellline = self.act(self.fc_cell2(x_cellline))
        return x_drug, x_cellline


class Syn_Prediction(torch.nn.Module):
    def __init__(self, in_channels):
        super(Syn_Prediction, self).__init__()
        self.MH = CrossAttention(2, 256, 0.5, 0.5, 1e-5)
        self.FW = FeedForward(256, 256, 0.5, 'sigmoid', 1e-5)

        self.fc1 = nn.Linear(in_channels, in_channels // 2)
        self.batch1 = nn.BatchNorm1d(in_channels // 2)
        self.fc2 = nn.Linear(in_channels // 2, in_channels // 4)
        self.batch2 = nn.BatchNorm1d(in_channels // 4)
        self.fc3 = nn.Linear(in_channels // 4, 1)
        self.drop_out = nn.Dropout(0.3)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, graph_embed, druga_id, drugb_id, cellline_id,drug,drug_num):

        cat_embed=torch.cat((drug,graph_embed[:drug_num]),1)
        #h = torch.cat((graph_embed[druga_id, :], graph_embed[drugb_id, :], graph_embed[cellline_id, :]), 1)


        h = torch.cat((cat_embed[druga_id, :], cat_embed[drugb_id, :], graph_embed[cellline_id, :]), 1)
        h = self.act(self.fc1(h))
        h = self.batch1(h)
        h = self.drop_out(h)
        h = self.act(self.fc2(h))
        h = self.batch2(h)
        h = self.drop_out(h)
        h = self.fc3(h)
        return h.squeeze(dim=1)


class MMHLSyn(torch.nn.Module):
    def __init__(self, bio_encoder, graph_encoder, Syn_Prediction):
        super(MMHLSyn, self).__init__()
        self.bio_encoder = bio_encoder
        self.graph_encoder = graph_encoder
        self.Syn_Prediction = Syn_Prediction
        self.li=nn.Linear(384,512)
        self.drug_rec_weight = nn.Parameter(torch.rand(1024, 1024))
        self.cline_rec_weight = nn.Parameter(torch.rand(512, 512))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.bio_encoder)
        reset(self.graph_encoder)
        reset(self.Syn_Prediction)

    def forward(self, drug_feature, drug_adj, ibatch, gexpr_data, adj, druga_id, drugb_id, cellline_id,drug):
        drug_embed, cellline_embed = self.bio_encoder(drug_feature, drug_adj, ibatch, gexpr_data)
        drug=self.li(drug)
        merge_embed = torch.cat((drug_embed, cellline_embed), 0)
        graph_embed = self.graph_encoder(merge_embed, adj)
        drug_emb, cline_emb = graph_embed[:38], graph_embed[38:]
    #药物smiles特征与药物交互特征拼接
        drug_emb = torch.cat((drug_emb,drug), 1)
    #Multi-task Module Based on Similarity
        #创建新的矩阵
        rec_drug = torch.sigmoid(torch.mm(torch.mm(drug_emb, self.drug_rec_weight), drug_emb.t()))
        rec_cline = torch.sigmoid(torch.mm(torch.mm(cline_emb, self.cline_rec_weight), cline_emb.t()))
        res = self.Syn_Prediction(graph_embed, druga_id, drugb_id, cellline_id,drug,38)
        return res, rec_drug, rec_cline
