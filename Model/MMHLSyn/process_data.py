#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# @Author: Scz 
# @Time:  2022/3/31 16:41
import deepchem as dc
from rdkit import Chem
import pandas as pd
import numpy as np
from utils import get_MACCS
from drug_util import drug_feature_extract
from sentence_transformers import SentenceTransformer



def getData(dataset):
    if  dataset == 'ALMANAC':
        drug_smiles_file = '../../Data/ALMANAC-COSMIC/drug_smiles.csv'
        cline_feature_file = '../../Data/ALMANAC-COSMIC/cell line_gene_expression.csv'
        drug_synergy_file = '../../Data/ALMANAC-COSMIC/drug_synergy.csv'
    if dataset=='DrugComb':
        drug_smiles_file = '../../Data/DrugComb/drug.csv'
        cline_feature_file = '../../Data/DrugComb/cell.csv'
        drug_synergy_file = '../../Data/DrugComb/drug_comb.csv'
    if dataset == 'DrugCombDB':
        drug_smiles_file = '../../Data/1-DrugCombDB/drug.csv'
        cline_feature_file = '../../Data/1-DrugCombDB/cell.csv'
        drug_synergy_file = '../../Data/1-DrugCombDB/drug_comb.csv'
    if dataset == 'oneil':
        drug_smiles_file = '../../Data/oneil/drug.csv'
        cline_feature_file = '../../Data/oneil/cell_qianliang.csv'
        drug_synergy_file = '../../Data/oneil/drug_comb.csv'
    gene_data = pd.read_csv(cline_feature_file, sep=',', header=0, index_col=[0])
    cline_num = len(gene_data.index)

    model_name = '../../FineBert/simcsesqrt-model'
    drug_model = SentenceTransformer(model_name)
    drug = pd.read_csv(drug_smiles_file, sep=',', header=0, index_col=[0])#读取文件返回一个DataFrame表格形式
    drug_data = pd.DataFrame()#DataFrame 是一个表格型的数据结构，它含有一组有序的列，每列可以是不同的值类型（数值、字符串、布尔型值）。DataFrame 既有行索引也有列索引，它可以被看做由 Series 组成的字典（共同用一个索引）。
    drug_smiles_fea = []#空列表
    drug_smiles_fea1=[]


    # 实现分子特征化
    featurizer = dc.feat.ConvMolFeaturizer()

    for tup in zip(drug['ID'], drug['isosmiles']):#把id和isosmiles组成元祖，作为列表的一个元素
        drug_smiles_fea1.append(drug_model.encode(tup[1][:128]))
        mol = Chem.MolFromSmiles(tup[1])#这行代码将所选元组的一个分子的 SMILES 表示法转换为一个分子对象（）
        mol_f = featurizer.featurize(mol)#对该分子对象进行分子特征化
        drug_data[str(tup[1])] = [mol_f[0].get_atom_features(), mol_f[0].get_adjacency_list()]
        drug_smiles_fea.append(get_MACCS(tup[1]))
    drug_num = len(drug_data.keys())
    d_map = dict(zip(drug_data.keys(), range(drug_num)))#字典
    drug_fea = drug_feature_extract(drug_data)#相当于对邻接信息进行了整理
    gene_data = pd.read_csv(cline_feature_file, sep=',', header=0, index_col=[0])
    cline_num = len(gene_data.index)
    c_map = dict(zip(gene_data.index, range(drug_num, drug_num + cline_num)))
    cline_fea = np.array(gene_data, dtype='float32')
    synergy_load = pd.read_csv(drug_synergy_file, sep=',', header=0)
    synergy = [[d_map[str(row[0])], d_map[str(row[1])], c_map[row[2]], float(row[3])] for index, row in
               synergy_load.iterrows() if (str(row[0]) in drug_data.keys() and str(row[1]) in drug_data.keys() and
                                           str(row[2]) in gene_data.index)]
    return cline_fea, drug_fea, drug_smiles_fea, drug_smiles_fea1,gene_data, synergy