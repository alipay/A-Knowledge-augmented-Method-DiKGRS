import numpy as np
import pickle as pkl
from collections import defaultdict
import os

def data_reform(dataset):
    if dataset == 'movie':
        train_pkl_path = 'data/movie/train_data.pkl'
        test_pkl_path = 'data/movie/test_data.pkl'
    if dataset =='last-fm':
        train_pkl_path = 'data/last-fm/train_data.pkl'
        test_pkl_path = 'data/last-fm/test_data.pkl'
    if dataset =='mybank1':
        train_pkl_path = 'data/mybank1/train_data.pkl'
        test_pkl_path = 'data/mybank1/test_data.pkl'
    if dataset =='mybank2':
        train_pkl_path = 'data/mybank2/train_data.pkl'
        test_pkl_path = 'data/mybank2/test_data.pkl'
    with open(train_pkl_path,'rb') as f_train:
        train_data_raw = pkl.load(f_train)
    train_table_ps = np.loadtxt('data/' + dataset + '/train_table.txt',dtype = np.int32)
    with open(test_pkl_path,'rb') as f_test:
        test_data_raw = pkl.load(f_test)
    v2w(train_data_raw,train_pkl_path)
    v2w(test_data_raw, test_pkl_path)
    v2w(train_table_ps,'data/' + dataset + '/train_table.txt')
    v2w_flag(train_table_ps,'data/' + dataset + '/train_table.txt')
    reform_kg_format(dataset)




def v2w(data,path):
    u2i_dict = defaultdict(list)
    for line in data:
        u2i_dict[line[0]].append(line[1])
    u2i_dict = {key:u2i_dict[key] for key in sorted(u2i_dict.keys())}
    if '_data.pkl' in path:
        wr_path = path.replace('_data.pkl','.txt')
    else:
        wr_path = path.replace('_table.txt','_ps.txt')
    with open(wr_path,'w') as f:
        for key, value_list in u2i_dict.items():
            items = ' '.join([str(item) for item in value_list])
            f.write(str(key)+' ' + items + '\n')

def v2w_flag(data, path):
    u2i_dict = defaultdict(list)
    for line in data:
        u2i_dict[line[0]].append(line[3])
    u2i_dict = {key:u2i_dict[key] for key in sorted(u2i_dict.keys())}
    if '_data.pkl' in path:
        wr_path = path.replace('_data.pkl','.txt')
    else:
        wr_path = path.replace('_table.txt','_ps_flag.txt')
    with open(wr_path,'w') as f:
        for key, value_list in u2i_dict.items():
            items = ' '.join([str(item) for item in value_list])
            f.write(str(key)+' ' + items + '\n')

def reform_kg_format(dataset):
    kg_path = 'data/'+ dataset + '/'+'kg_final.txt'
    can_triplets_np = np.loadtxt(kg_path, dtype=np.int32)
    with open('data/'+ dataset + '/'+'kg_final_1.txt','w') as fw:
        for line in can_triplets_np:
            fw.write('{} {} {}\n'.format(str(line[0]),str(line[1]),str(line[2])))
    print('done')


if __name__ == '__main__':
    data_reform('movie')
    # data_reform('last-fm')
    # reform_kg_format('movie')
