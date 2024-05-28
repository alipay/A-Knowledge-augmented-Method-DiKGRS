'''
Created on Nov 11, 2023
PyTorch Implementation of dataloader_DiKGRS
@author: Xuying Ning
'''
__author__ = "Xuying Ning"

import os
import random
from typing import DefaultDict
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from collections import defaultdict
import json
from itertools import chain
import pickle
import networkx as nx
import scipy.sparse as sp

from collections import defaultdict
import warnings
import torch.nn.functional as F
warnings.filterwarnings('ignore')

n_users = 0
n_items = 0
n_entities = 0
n_relations = 0
n_nodes = 0
train_user_set = defaultdict(list)
test_user_set = defaultdict(list)

# one-hot encode in dataset get item process
def map_score(value,low_sep,mid_sep,high_sep):
    if value < low_sep:
        return 0
    if value >= low_sep and value < mid_sep:
        return 1
    if value >= mid_sep and value < high_sep:
        return 2
    if value >= high_sep:
        return 3


def popularity_encoding(rating_all):
    exp_pv = dict()
    clk_pv = dict()
    for line in rating_all:
        item = line[1]
        if item not in exp_pv:
            exp_pv[item] = 1
        else:
            exp_pv[item] += 1
        if item not in clk_pv:
            clk_pv[item] = int(line[2])
        else:
            clk_pv[item] += int(line[2])
    high_sep = np.percentile(list(exp_pv.values()),75)
    exp_encoded =  {k:int(v >= high_sep) for k,v  in exp_pv.items()}
    high_sep_clk = np.percentile(list(clk_pv.values()),75)
    clk_encoded =  {k:int(v >= high_sep_clk ) for k,v  in clk_pv.items()}
    return exp_encoded,clk_encoded

def user_profile_encoding(rating_all):
    u_profile_1 = list(set(rating_all[:,3]))
    u_profile_2 = list(set(rating_all[:,4]))
    u_profile_3 = list(set(rating_all[:,5]))
    user_field = len(u_profile_1)+len(u_profile_2) + len(u_profile_3)
    item_up_encode = dict()
    for line in rating_all:
        item = line[1]
        up_1,up_2,up_3,up_4 = line[3],line[4],line[5],line[6]
        if item not in item_up_encode:
            item_up_encode[item] = [0] * user_field
        item_up_encode[item][u_profile_1.index(up_1)] += int(line[2])
        item_up_encode[item][len(u_profile_1)+u_profile_2.index(up_2)] += int(line[2])
        item_up_encode[item][len(u_profile_1)+len(u_profile_2)+u_profile_3.index(up_3)] += int(line[2])
    

    new_item_up_encode = dict()
    for i in range(user_field):
        high_sep_up = np.percentile([v[i] for p,v in item_up_encode.items()], 75)
        for item, values in item_up_encode.items():
            if item not in new_item_up_encode:
                new_item_up_encode[item] = values.copy()
            new_item_up_encode[item][i] = int(values[i] >= high_sep_up )
    return new_item_up_encode


def user_profile_encoding_mybank(data_path,rating_all,user_fea_dict):
    # rating_all = np.loadtxt(rating_all_fname, dtye = np.int32)
    num_fea = len(list(user_fea_dict.values())[0])
    new_rating_all = []
    for line in rating_all:
        user = line[0]
        # print(line, user_fea_dict[user])
        new_rating_all.append(list(line) + list(user_fea_dict[user]))
    
    new_rating_all = np.array(new_rating_all)

    user_field = 0
    user_profile_i_dict = dict()
    for fea_id in range(3,num_fea+3):
        # print(new_rating_all)
        # print(fea_id)
        # print(set(new_rating_all[:,fea_id]))
        u_profile_i = list(set(new_rating_all[:,fea_id]))
        user_field += len(u_profile_i)
        user_profile_i_dict[fea_id-3] = u_profile_i
    
    print(user_field)
    # print(user_profile_i_dict)
    
    item_up_encode = dict()
    item_clk_rate = {item : np.sum(new_rating_all[:,1]==item) / len(new_rating_all) for item in np.unique(new_rating_all[:,1])}

    for line in new_rating_all:
        item = line[1]
        up_list = line[3:]
        if item not in item_up_encode:
            item_up_encode[item] = [0] * user_field
        for up_id in range(len(up_list)):
            prev_len = sum([len(user_profile_i_dict[i]) for i in range(up_id)])
            item_up_encode[item][prev_len + user_profile_i_dict[up_id].index(up_list[up_id])] += int(line[2])
    # print(item_up_encode[16])
    
    all_user = len(user_fea_dict.keys())
    user_crowd_num = dict()
    for uid,value in user_fea_dict.items():
        for up_id in range(len(value)):
            prev_len = sum([len(user_profile_i_dict[i]) for i in range(up_id)])
            user_crowd = prev_len + user_profile_i_dict[up_id].index(value[up_id])
            if user_crowd not in user_crowd_num:
                user_crowd_num[user_crowd] = 0
            user_crowd_num[user_crowd] += 1
    
    
    user_crowd_rate = {crowd : user_crowd_num[crowd] / all_user for crowd in user_crowd_num.keys()}
    # print('num crowds:', len(user_crowd_rate.keys()))
    # print(sum(user_crowd_rate.values()))
    # print(len(list(user_fea_dict.values())[0]))

    new_item_up_encode = dict()
    for i in range(user_field):
        user_field_all_user = sum([item_up_encode[item][i] for item in item_up_encode.keys()])
        for item, value in item_up_encode.items():
            if item not in new_item_up_encode:
                new_item_up_encode[item] = [0] * user_field
            new_item_up_encode[item][i] = int( (value[i] / user_field_all_user) / item_clk_rate[item] >= 1.5)

    with open(data_path + 'item_feature_table.txt', 'w') as fw:
        for iid in new_item_up_encode.keys():
            fw.write(str(int(iid)) + ' ' + ' '.join([str(c) for c in new_item_up_encode[iid]])+'\n')
    return new_item_up_encode



#正负样本采样在这进行
def get_train_test_data(rating_all_fname,pseudo_rating_fname,test_pkl_fname, path, num_ps,dataset, backbone):
    if dataset != 'mybank1' and dataset != 'mybank2':
        #adding real or pseudo label
        ratings = np.loadtxt(rating_all_fname,dtype = np.int32)
        dataset_fname = dataset + '/'
        """item feature encoding"""
        exp_encoded,clk_encoded = popularity_encoding(ratings)
        if dataset == 'movie':
            item_up_encode = user_profile_encoding(ratings)

        ratings = np.concatenate((ratings,np.array([1]*ratings.shape[0]).reshape(-1,1)),axis = 1)
        with open(test_pkl_fname,'rb') as f:
            test_data = pickle.load(f, encoding='bytes')
        print(test_data.shape)
        np.savetxt(path+dataset_fname+'test_table.txt', test_data.astype(int), delimiter=' ', fmt='%d')
        print('test_table generated')

        n_items = max(max(ratings[:,1]),max(test_data[:,1])) + 1

        pseudo_rating = np.loadtxt(pseudo_rating_fname, dtype = np.int32)
        pseudo_rating = pseudo_rating[np.random.randint(0,len(pseudo_rating),num_ps)]
        pseudo_rating = np.concatenate((pseudo_rating,np.array([0]*pseudo_rating.shape[0]).reshape(-1,1)),axis = 1)
        print('Number of Pseudo Samples:', len(pseudo_rating))
        rating_all = np.concatenate((ratings, pseudo_rating), axis=0)
        # uid, iid, label, rp_label
        rating_all = np.concatenate((rating_all[:,:3].copy(),rating_all[:,-1].reshape(-1,1).copy()), axis = 1)
        pos_training = rating_all[rating_all[:,2] == 1]
        print(pos_training.shape)

        np.savetxt(path+dataset_fname+'train_table.txt', pos_training.astype(int), delimiter=' ', fmt='%d')
        print('training_table generated')

        if dataset == 'movie':
            if len(list(exp_encoded.keys())) != len(list(item_up_encode.keys())):
                print('unmatch items!')
        item_feature_dict = defaultdict(list)
        for item in range(n_items):
            if item not in clk_encoded:
                exp_encoded[item],clk_encoded[item] = 0,0
                if dataset == 'movie':
                    item_up_encode[item] = [0] * len(list(item_up_encode.values())[0])
            if dataset == 'movie':
                item_feature = [exp_encoded[item]] + [clk_encoded[item]] + item_up_encode[item]
            else:
                item_feature = [exp_encoded[item]] + [clk_encoded[item]]
            item_feature_dict[item] = item_feature

        with open(path+dataset_fname+'/item_feature_table.txt','w') as f:
            for item in item_feature_dict:
                f.write(str(item) + ' ' + ' '.join(str(x) for x in item_feature_dict[item]) + '\n')
        print('item_feature_table generated')

    
    else:
        dataset_fname = dataset + '/'
        # id_dict, new_entity_lookup = get_id_dict(path + dataset_fname, backbone)
        # train_table, user_dict = generate_train_test_table_mybank(path + dataset_fname, id_dict,)
        # u_feature_dict_new_hashed = get_other_txt_mybank(path + dataset_fname, new_entity_lookup, user_dict, backbone)
        u_feature_dict_new_hashed_array = np.loadtxt(path + dataset_fname + 'users.txt', delimiter = '::')
        u_feature_dict_new_hashed = {int(line[0]) : line[1:].tolist() for line in u_feature_dict_new_hashed_array}
        ratings = np.loadtxt(rating_all_fname, dtype = np.int32)
        ratings_new = np.concatenate((ratings,np.array([1]*ratings.shape[0]).reshape(-1,1)),axis = 1)
        pseudo_rating = np.loadtxt(pseudo_rating_fname, dtype = np.int32)
        pseudo_rating = pseudo_rating[np.random.randint(0,len(pseudo_rating),num_ps)]
        pseudo_rating = np.concatenate((pseudo_rating,np.array([0]*pseudo_rating.shape[0]).reshape(-1,1)),axis = 1)
        print('Number of Pseudo Samples:', len(pseudo_rating))
        rating_all = np.concatenate((ratings_new, pseudo_rating), axis=0)
        rating_all = np.concatenate((rating_all[:,:3].copy(),rating_all[:,-1].reshape(-1,1).copy()), axis = 1)
        pos_training = rating_all[rating_all[:,2] == 1]
        print(pos_training.shape)

        np.savetxt(path+dataset_fname +'/train_table.txt', pos_training.astype(int), delimiter=' ', fmt='%d')
        print('training_table generated')


        new_item_up_encode = user_profile_encoding_mybank(path + dataset_fname,ratings,u_feature_dict_new_hashed)
        print('item feature table generated')

    return

def load_rating(args):
    print('reading training file and testing file ...')
    directory = 'data/' + args.dataset + '/'
    if args.add_pseudo:
        train_data  = np.loadtxt(directory + 'train_table.txt', dtype = np.int32)
    else:
        with open(directory + "/train_data.pkl", 'rb') as ft:
            train_data = pickle.load(ft, encoding='bytes')

    with open(directory + "/test_data.pkl", 'rb') as fi:
        test_data = pickle.load(fi, encoding='bytes')
    rating_np = np.concatenate((train_data[:,:3].copy(), test_data), axis=0)

    # reading rating file
    n_user = max(set(rating_np[:, 0])) + 1  # the result = max(rating_np[:, 0]
    n_item = max(set(rating_np[:, 1])) + 1  # the result = max(rating_np[:, 1]

    return n_user, n_item, train_data, test_data


def load_data(model_args):
    global args
    args = model_args
    directory = args.data_path + args.dataset + '/'

    print('reading train and test user-item set ...')
    n_user, n_item, train_data_raw, test_data = load_rating(args)

    train_data = train_data_raw[:,:3].copy()

    train_cf = read_cf(train_data)
    test_cf = read_cf(test_data)
    remap_item(train_cf, test_cf)

    pseudo_flag_dict = dict()
    for line in train_data_raw:
        uid, iid = line[0], line[1]
        if args.add_pseudo:
            pseudo_flag = line[3]
            pseudo_flag_dict[(uid,iid)] = pseudo_flag
        else:
            pseudo_flag_dict[(uid,iid)] = 1

    print('combinating train_cf and kg data ...')
    if args.dataset == 'mybank1' and (args.backbone == 'VRKG' or args.backbone == 'KGIN'):
            triplets = read_triplets(directory + 'kg_final_cut.txt')
    else:
        triplets = read_triplets(directory + 'kg_final.txt')

    print('building the graph ...')
    graph, relation_dict = build_graph(train_cf, triplets)

    print('building the adj mat ...')
    adj_mat_list, norm_mat_list, mean_mat_list = build_sparse_relational_graph(relation_dict)

    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
        'n_entities': int(n_entities),
        'n_nodes': int(n_nodes),
        'n_relations': int(n_relations)
    }

    user_dict = {
        'train_user_set': train_user_set,
        'test_user_set': test_user_set
    }

    item_feature_data = np.loadtxt(directory + 'item_feature_table.txt', dtype = np.int32)
    item_feature_dict = dict()
    for line in item_feature_data:
        item_feature_dict[line[0]] = line[1:].copy()

    return train_cf, test_cf, user_dict, n_params, graph, triplets, relation_dict,\
           [adj_mat_list, norm_mat_list, mean_mat_list], pseudo_flag_dict, item_feature_dict


def read_cf(data_name):
    inter_mat = data_name[:, :2]

    return np.array(inter_mat)


def remap_item(train_data, test_data):
    global n_users, n_items
    n_users = max(max(train_data[:, 0]), max(test_data[:, 0])) + 1
    n_items = max(max(train_data[:, 1]), max(test_data[:, 1])) + 1

    for u_id, i_id in train_data:
        train_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in test_data:
        test_user_set[int(u_id)].append(int(i_id))


def read_triplets(file_name):
    global n_entities, n_relations, n_nodes

    can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
    can_triplets_np = np.unique(can_triplets_np, axis=0)

    if 1:
        # get triplets with inverse direction like <entity, is-aspect-of, item>
        inv_triplets_np = can_triplets_np.copy()
        inv_triplets_np[:, 0] = can_triplets_np[:, 2]
        inv_triplets_np[:, 2] = can_triplets_np[:, 0]
        inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1
        # consider two additional relations --- 'interact' and 'be interacted'
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        inv_triplets_np[:, 1] = inv_triplets_np[:, 1] + 1
        # get full version of knowledge graph
        triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)
    else:
        # consider two additional relations --- 'interact'.
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        triplets = can_triplets_np.copy()

    n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1  # including items + users
    n_nodes = n_entities + n_users
    n_relations = max(triplets[:, 1]) + 1

    return triplets


def build_graph(train_data, triplets):
    ckg_graph = nx.MultiDiGraph()
    rd = defaultdict(list)

    print("Begin to load interaction triples ...")
    for u_id, i_id in tqdm(train_data, ascii=True):
        rd[0].append([u_id, i_id])

    print("\nBegin to load knowledge graph triples ...")
    for h_id, r_id, t_id in tqdm(triplets, ascii=True):
        ckg_graph.add_edge(h_id, t_id, key=r_id)
        rd[r_id].append([h_id, t_id])

    return ckg_graph, rd


def build_sparse_relational_graph(relation_dict):
    def _bi_norm_lap(adj):
        # D^{-1/2}AD^{-1/2}
        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def _si_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    adj_mat_list = []
    print("Begin to build sparse relation matrix ...")
    for r_id in tqdm(relation_dict.keys()):
        np_mat = np.array(relation_dict[r_id])
        if r_id == 0:
            cf = np_mat.copy()
            cf[:, 1] = cf[:, 1] + n_users  # [0, n_items) -> [n_users, n_users+n_items)
            vals = [1.] * len(cf)
            adj = sp.coo_matrix((vals, (cf[:, 0], cf[:, 1])), shape=(n_nodes, n_nodes))
        else:
            vals = [1.] * len(np_mat)
            adj = sp.coo_matrix((vals, (np_mat[:, 0], np_mat[:, 1])), shape=(n_nodes, n_nodes))
        adj_mat_list.append(adj)

    norm_mat_list = [_bi_norm_lap(mat) for mat in adj_mat_list]
    mean_mat_list = [_si_norm_lap(mat) for mat in adj_mat_list]
    # interaction: user->item, [n_users, n_entities]
    norm_mat_list[0] = norm_mat_list[0].tocsr()[:n_users, n_users:].tocoo()
    mean_mat_list[0] = mean_mat_list[0].tocsr()[:n_users, n_users:].tocoo()

    return adj_mat_list, norm_mat_list, mean_mat_list

    
# if __name__ == '__main__':
#     get_train_test_data('/ossfs/workspace/data/mybank1/rating_all.txt','/ossfs/workspace/data/mybank1/pseudo_ratings.txt','/ossfs/workspace/data/mybank1/test_data.pkl', path ='/ossfs/workspace/data/' , num_ps = 3846, dataset = 'mybank1', backbone = 'VRKG')