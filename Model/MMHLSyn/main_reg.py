#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# @Author: Scz 
# @Time:  2021/12/9 16:28

import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from model_reg import Atom_Gloal, MMHLSyn, Interactive, Syn_Prediction
from sklearn.model_selection import KFold
import os
import glob
import warnings
import sys
sys.path.append('..')

from drug_util import GraphDataset, collate
from utils import regression_metric, set_seed_all
from similarity import get_Cosin_Similarity, get_pvalue_matrix
from process_data import getData

warnings.filterwarnings('ignore')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')#'cuda:0' if torch.cuda.is_available() else
print(device)


def load_data(dataset):
    cline_fea, drug_fea, drug_smiles_fea, drug,gene_data, synergy= getData(dataset)
    cline_fea = torch.from_numpy(cline_fea).to(device)
    X = np.mat(drug)
    X=np.array(X)
    drug=torch.from_numpy(X).type(torch.FloatTensor).to(device)
    drug_sim_matrix, cline_sim_matrix = get_sim_mat(drug_smiles_fea, np.array(gene_data, dtype='float32'))

    return drug_fea, cline_fea, synergy, drug_sim_matrix, cline_sim_matrix,drug


def data_split(synergy,rd_seed=0):
    synergy_pos = pd.DataFrame([i for i in synergy])
    # -----split synergy into 5CV,test set
    train_size = 1
    synergy_cv_data, synergy_test = np.split(np.array(synergy_pos.sample(frac=1, random_state=rd_seed)),
                                             [int(train_size * len(synergy_pos))])
    np.random.shuffle(synergy_cv_data)
    np.savetxt(path + 'test_y_true.txt', synergy_test[:, 3])
    np.savetxt(path + 'test_y_true_111.txt', synergy_test)
    test_label = torch.from_numpy(np.array(synergy_test[:, 3], dtype='float32')).to(device)
    test_ind = torch.from_numpy(np.array(synergy_test[:, 0:3])).long().to(device)
    return synergy_cv_data, test_ind, test_label


def get_sim_mat(drug_fea, cline_fea):
    drug_sim_matrix = np.array(get_Cosin_Similarity(drug_fea))
    cline_sim_matrix = np.array(get_pvalue_matrix(cline_fea))
   # drug_sim_matrix = att_sim(drug_fea).to(device)
    #cline_sim_matrix = att_sim(cline_fea).to(device)
   # return drug_sim_matrix,cline_sim_matrix
    return torch.from_numpy(drug_sim_matrix).type(torch.FloatTensor).to(device), torch.from_numpy(
         cline_sim_matrix).type(torch.FloatTensor).to(device)


# --train+test
def train(drug_fea_set, cline_fea_set, synergy_adj, index, label, alpha,drug_):
    loss_train = 0
    true_ls, pre_ls = [], []
    optimizer.zero_grad()

    for batch, (drug, cline) in enumerate(zip(drug_fea_set, cline_fea_set)):
        pred, rec_drug, rec_cline= model(drug.x.to(device), drug.edge_index.to(device), drug.batch.to(device),
                                          cline[0], synergy_adj,
                                          index[:, 0], index[:, 1], index[:, 2],drug_)
        loss = loss_func(pred, label)
        loss_rec_2 = rec_loss(rec_cline, cline_sim_mat)
        loss_rec_1 = rec_loss(rec_drug, drug_sim_mat)
        loss = (1 - alpha) * loss + alpha * (loss_rec_1 + loss_rec_2)
        loss.backward()
       # loss.backward(retain_graph=True)

        optimizer.step()

        loss_train += loss.item()
        true_ls += label_train.cpu().detach().numpy().tolist()
        pre_ls += pred.cpu().detach().numpy().tolist()
    rmse, r2, pr,mae,rs,mse = regression_metric(true_ls, pre_ls)
    return [rmse, r2, pr,mae,rs,mse], loss_train


def test(drug_fea_set, cline_fea_set, synergy_adj, index, label, alpha,drug_):
    model.eval()
    with torch.no_grad():
        for batch, (drug, cline) in enumerate(zip(drug_fea_set, cline_fea_set)):
            pred, rec_drug, rec_cline= model(drug.x.to(device), drug.edge_index.to(device), drug.batch.to(device),
                                              cline[0], synergy_adj,
                                              index[:, 0], index[:, 1], index[:, 2],drug_)
        loss = loss_func(pred, label)
        loss_rec_1 = loss_func(rec_drug, drug_sim_mat)
        loss_rec_2 = loss_func(rec_cline, cline_sim_mat)
        loss = (1 - alpha) * loss + alpha * (loss_rec_1 + loss_rec_2)
        rmse, r2, pr,mae,rs,mse = regression_metric(label.cpu().detach().numpy(),
                                         pred.cpu().detach().numpy())
        return [rmse, r2, pr,mae,rs,mse], loss.item(), pred.cpu().detach().numpy()


if __name__ == '__main__':
    dataset_name = 'oneil'    # or ONEIL  ALMANAC-COSMIC' DrugComb DrugCombDB
    seed = 0
    cv_mode_ls =[3]
    epochs = 2
    learning_rate = 4e-3
    L2 = 1e-3
    alpha = 0.4
    a=[]
    for cv_mode in cv_mode_ls:
        path = 'result_reg' + dataset_name + '_' + str(cv_mode) + '_'
        file = open(path + 'result.txt', 'w')
        set_seed_all(seed)
        drug_feature, cline_feature, synergy_data, drug_sim_mat, cline_sim_mat,drug = load_data(dataset_name)
        drug_set = Data.DataLoader(dataset=GraphDataset(graphs_dict=drug_feature),
                                   collate_fn=collate, batch_size=len(drug_feature), shuffle=False)
        cline_set = Data.DataLoader(dataset=Data.TensorDataset(cline_feature),
                                    batch_size=len(cline_feature), shuffle=False)
        # %%-----build dataset
        # -----split synergy into 5CV,test set
        synergy_cv, index_test, label_test = data_split(synergy_data)
        if cv_mode == 1:
            cv_data = synergy_cv
        elif cv_mode == 2:
            cv_data = np.unique(synergy_cv[:, 2])  # cline_level
        elif cv_mode == 4:
            for i in range(38):
                a.append(i)
            a=np.unique(a)
          #  cv_data = np.unique(synergy_cv[:, 1])#drug_level
            cv_data=a
            # cline_level
        else:
            cv_data = np.unique(np.vstack([synergy_cv[:, 0], synergy_cv[:, 1]]), axis=1).T  # drug pairs_level
        # ---5CV
        MSEL=np.zeros(5)
        P=np.zeros(5)
        RMSEL=np.zeros(5)
        final_metric = np.zeros(6)
        fold_num = 0
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        for train_index, validation_index in kf.split(cv_data):
            # ---construct train_set+validation_set
            if cv_mode == 1:  # normal_level
                synergy_train, synergy_validation = cv_data[train_index], cv_data[validation_index]
            elif cv_mode == 2:  # cell line_level
                train_name, test_name = cv_data[train_index], cv_data[validation_index]
                synergy_train = np.array([i for i in synergy_cv if i[2] in train_name])
                synergy_validation = np.array([i for i in synergy_cv if i[2] in test_name])
            elif cv_mode==4:#drugs_level
                train_name, test_name = cv_data[train_index], cv_data[validation_index]
                synergy_train = np.array([i for i in synergy_cv if (i[0] in train_name) and (i[1] in train_name)] )
                synergy_validation = np.array([i for i in synergy_cv if (i[0] in test_name) or (i[1] in test_name)])
            else:  # drug pairs_level
                pair_train, pair_validation = cv_data[train_index], cv_data[validation_index]
                synergy_train = np.array(
                    [j for i in pair_train for j in synergy_cv if (i[0] == j[0]) and (i[1] == j[1])])
                synergy_validation = np.array(
                    [j for i in pair_validation for j in synergy_cv if (i[0] == j[0]) and (i[1] == j[1])])
            # ---construct train_set+validation_set
            np.savetxt(path + 'val_' + str(fold_num) + '_true.txt', synergy_validation[:, 3])
            label_train = torch.from_numpy(np.array(synergy_train[:, 3], dtype='float32')).to(device)
            label_validation = torch.from_numpy(np.array(synergy_validation[:, 3], dtype='float32')).to(device)
            index_train = torch.from_numpy(synergy_train[:, 0:3]).long().to(device)#获取synergy——train的前三行
            index_validation = torch.from_numpy(synergy_validation[:, 0:3]).long().to(device)
            # -----construct hyper_synergy_graph_set
            synergy_train_tmp = np.copy(synergy_train)
            for data in synergy_train_tmp:
                data[3] = 1 if data[3] >= 30 else 0
            pos_edge = np.array([t for t in synergy_train_tmp if t[3] != 0])
            edge_data = pos_edge[:, 0:3]
            synergy_edge = edge_data.reshape(1, -1)
            index_num = np.expand_dims(np.arange(len(edge_data)), axis=-1)
            synergy_num = np.concatenate((index_num, index_num, index_num), axis=1).reshape(1, -1)
            synergy_graph = np.concatenate((synergy_edge, synergy_num), axis=0)
            synergy_graph = torch.from_numpy(synergy_graph).type(torch.LongTensor).to(device)

            # ---model_build
#2560
            model = MMHLSyn(Atom_Gloal(dim_drug=75, dim_cellline=cline_feature.shape[-1], output=100),
                                      Interactive(in_channels=100, out_channels=512), Syn_Prediction(in_channels=2560)).to(device)
            loss_func = torch.nn.MSELoss()
            rec_loss = torch.nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L2)

            best_metric = [0, 0, 0]
            best_epoch = 0
            j=0
            for epoch in range(epochs):
                model.train()
                train_metric, train_loss = train(drug_set, cline_set, synergy_graph,
                                                 index_train, label_train, alpha,drug)
                val_metric, val_loss, _ = test(drug_set, cline_set, synergy_graph,
                                               index_validation, label_validation, alpha,drug)
                if epoch % 20 == 0:
                    print('Epoch: {:05d},'.format(epoch), 'loss_train: {:.6f},'.format(train_loss),
                          'RMSE: {:.6f},'.format(train_metric[0]), 'R2: {:.6f},'.format(train_metric[1]),
                          'Pearson r: {:.6f},'.format(train_metric[2]),
                          'mae: {:.6f},'.format(train_metric[3]),
                          'pr: {:.6f},'.format(train_metric[4]),
                          'mse: {:.6f},'.format(train_metric[5]))
                    print('Epoch: {:05d},'.format(epoch), 'loss_val: {:.6f},'.format(val_loss),
                          'RMSE: {:.6f},'.format(val_metric[0]), 'R2: {:.6f},'.format(val_metric[1]),
                          'Pearson r: {:.6f},'.format(val_metric[2]),
                    'mae: {:.6f},'.format(val_metric[3]),
                    'pr: {:.6f},'.format(val_metric[4]),
                    'mse: {:.6f},'.format(val_metric[5])

                    )
                torch.save(model.state_dict(), '{}.pth'.format(epoch))
                if val_metric[2] > best_metric[2]:
                    best_metric = val_metric
                    best_epoch = epoch
                    j=0
                else:
                    j=j+1
                    if j>2000:
                        break
                files = glob.glob('*.pth')
                for f in files:
                    epoch_nb = int(f.split('.')[0])
                    if epoch_nb != best_epoch:
                        os.remove(f)
            # files = glob.glob('*.pth')
            # for f in files:
            #     epoch_nb = int(f.split('.')[0])
            #     if epoch_nb > best_epoch:
            #         os.remove(f)
            print('The best results on validation set, Epoch: {:05d},'.format(best_epoch),
                  'RMSE: {:.6f},'.format(best_metric[0]),
                  'R2: {:.6f},'.format(best_metric[1]), 'Pearson r: {:.6f},'.format(best_metric[2]),
                  'mae: {:.6f},'.format(best_metric[3]),
                  'pr: {:.6f},'.format(best_metric[4]),
                  'mse: {:.6f},'.format(best_metric[5])
                  )
            model.load_state_dict(torch.load('{}.pth'.format(best_epoch)))
            val_metric, _, y_val_pred = test(drug_set, cline_set, synergy_graph, index_validation,
                                             label_validation, alpha,drug)
            # test_metric, _, y_test_pred = test(drug_set, cline_set, synergy_graph, index_test,
            #                                     label_test, alpha,drug)

            # test_metric, _, y_test_pred=0,0,0
            # np.savetxt(path + 'val_' + str(fold_num) + '_pred.txt', y_val_pred)
            # np.savetxt(path + 'test_' + str(fold_num) + '_pred.txt', y_test_pred)

            file.write('val_metric:')
            for item in val_metric:
                file.write(str(item) + '\t')

            # file.write('\ntest_metric:')
            # for item in test_metric:
            #     file.write(str(item) + '\t')
            # file.write('\n')

            final_metric += val_metric
            RMSEL[fold_num]=val_metric[0]
            P[fold_num]=val_metric[2]
            MSEL[fold_num]=val_metric[5]
            #
            # final_metric += test_metric
            # RMSEL[fold_num] = test_metric[0]
            # P[fold_num] = test_metric[2]
            # MSEL[fold_num] = test_metric[5]

            fold_num = fold_num + 1
        final_metric /= 5

        def calc_stat(numbers):
            mu = sum(numbers) / len(numbers)
            sigma = (sum([(x - mu) ** 2 for x in numbers]) / len(numbers)) ** 0.5
            return mu, sigma


        def conf_inv(mu, sigma, n):
            delta = 2.776 * sigma / (n ** 0.5)  # 95%
            return mu - delta, mu + delta


        RMSE, RMSE_sig = calc_stat(RMSEL)
        MSE, MSE_sig = calc_stat(MSEL)
        p, p_sig = calc_stat(P)
        lo, hi = conf_inv(MSE, MSE_sig, 5)

        print('Final 5-cv average results, RMSE: {:.6f},'.format(final_metric[0]),
              'R2: {:.6f},'.format(final_metric[1]),
              'Pearson r: {:.6f},'.format(final_metric[2]),
              'mae: {:.6f},'.format(final_metric[3]),
              'pr: {:.6f},'.format(final_metric[4]),
              'mse: {:.6f},'.format(final_metric[5])
        )
        print('Final 5-cv average results, RMSE: {:.6f},+ RMSE_sig: {:.6f}'.format(RMSE,RMSE_sig),
              'Pearson r:{:.6f} + P_sig: {:.6f},'.format(p,p_sig),
              'mse: {:.6f}+ MSE_sig: {:.6f},'.format(MSE,MSE_sig),
              "Confidence interval: [{:.4f}, {:.4f}]".format(lo, hi)

        )

