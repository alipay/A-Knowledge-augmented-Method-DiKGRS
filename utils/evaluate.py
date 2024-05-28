from .metrics import *
from .parser import parse_args

import torch
import numpy as np
import multiprocessing
import heapq
from tqdm import tqdm
from collections import defaultdict
# from modules.VRKG_DVN import generate_item_batch_ratings

cores = multiprocessing.cpu_count() // 2

args = parse_args()
Ks = eval(args.Ks)
device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")
BATCH_SIZE = args.test_batch_size
batch_test_flag = args.batch_test_flag


def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc

def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)

    auc = AUC(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc

def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio, f1 = [], [], [], [], []

    for K in Ks:
        precision.append(precision_at_k(r, K))
        recall.append(recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(ndcg_at_k(r, K, user_pos_test))
        hit_ratio.append(hit_at_k(r, K))
        f1.append(F1(precision[0], recall[0]))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc,
            'f1': np.array(f1)}


def test_one_user(x, train_user_set, test_user_set):
    res = []
    for item in x:
        rating = item[0]
        u = item[1]
        # user u's items in the training set
        try:
            training_items = train_user_set[u]

        except Exception:
            training_items = []
        # user u's items in the test set
        user_pos_test = test_user_set[u]

        all_items = set(range(0, n_items))

        test_items = list(all_items - set(training_items))

        if args.test_flag == 'part':
            r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
        else:
            r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

        res.append(get_performance(user_pos_test, r, auc, Ks))

    return res

"""user dict can be all positive dict[user] = [all positive items]"""
def test(model, user_dict, n_params, item_feature_dict):
    result = {'precision': np.zeros(len(Ks)),
              'recall': np.zeros(len(Ks)),
              'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)),
              'auc': np.zeros(len(Ks)),
              'f1':np.zeros(len(Ks))}

    global n_users, n_items
    n_items = n_params['n_items']
    n_users = n_params['n_users']

    # global train_user_set, test_user_set, leack_items
    global leack_items
    train_user_set = user_dict['train_user_set']
    test_user_set = user_dict['test_user_set']

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE
    i_batch_size = BATCH_SIZE

    test_users = list(test_user_set.keys()) #all test_user ids, 0,1,2,3...
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0


    '''item_ids = item_batch['item_ids']
        item_batch_feature = item_batch['item_batch_feature']'''
    item_batch_inp = dict()
    item_batch_inp['item_ids'] = list()
    item_batch_inp['item_batch_feature'] = list()

    for iid in list(item_feature_dict.keys()):
        item_batch_inp['item_ids'].append(int(iid))
        item_batch_inp['item_batch_feature'].append(item_feature_dict[iid])
    
    item_batch_inp['item_ids'] = torch.tensor(item_batch_inp['item_ids']).long().to(device)
    item_batch_inp['item_batch_feature'] = torch.tensor(np.array(item_batch_inp['item_batch_feature'])).float().to(device)

    print(item_batch_inp['item_ids'].shape)
    print(item_batch_inp['item_batch_feature'].shape)

    item_all_emb, user_gcn_emb = model.new_generate(item_batch_inp)


    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = min((u_batch_id + 1) * u_batch_size, n_test_users)

        user_list_batch = test_users[start: end]
        user_batch = torch.LongTensor(np.array(user_list_batch)).to(device)
        u_g_embeddings = user_gcn_emb[user_batch]

        if batch_test_flag:
            # batch-item test
            n_item_batchs = n_items // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), n_items))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, n_items)

                item_batch = torch.LongTensor(np.array(range(i_start, i_end))).view(i_end-i_start).to(device)
                i_g_embddings = item_all_emb[item_batch]

                i_rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach().cpu()

                rate_batch[:, i_start: i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == n_items
        else:
            # all-item test
            item_batch = torch.LongTensor(np.array(range(0, n_items))).view(n_items, -1).to(device)
            i_g_embddings = item_all_emb[item_batch]
            rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach().cpu()

        user_batch_rating_uid = zip(rate_batch, user_list_batch)

        batch_result = test_one_user(user_batch_rating_uid, train_user_set, test_user_set)
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users
            result['auc'] += re['auc']/n_test_users
            result['f1'] += re['f1'] / n_test_users

    assert count == n_test_users
    pool.close()
    return result

def score(model, user_dict, n_params, item_feature_dict,out_dir):
    result = {'precision': np.zeros(len(Ks)),
              'recall': np.zeros(len(Ks)),
              'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)),
              'auc': np.zeros(len(Ks)),
              'f1':np.zeros(len(Ks))}

    global n_users, n_items
    n_items = n_params['n_items']
    n_users = n_params['n_users']

    # global train_user_set, test_user_set, leack_items
    global leack_items
    train_user_set = user_dict['train_user_set']
    test_user_set = user_dict['test_user_set']
    # all_user_set = user_dict

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE
    i_batch_size = BATCH_SIZE

    all_users = sorted(set(list(test_user_set.keys())).union(set(list(train_user_set.keys())))) #all test_user ids, 0,1,2,3...
    print(len(all_users))
    n_test_users = len(all_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0


    '''item_ids = item_batch['item_ids']
        item_batch_feature = item_batch['item_batch_feature']'''
    item_batch_inp = dict()
    item_batch_inp['item_ids'] = list()
    item_batch_inp['item_batch_feature'] = list()

    for iid in list(item_feature_dict.keys()):
        item_batch_inp['item_ids'].append(int(iid))
        item_batch_inp['item_batch_feature'].append(item_feature_dict[iid])
    
    item_batch_inp['item_ids'] = torch.tensor(item_batch_inp['item_ids']).long().to(device)
    item_batch_inp['item_batch_feature'] = torch.tensor(np.array(item_batch_inp['item_batch_feature'])).float().to(device)

    print(item_batch_inp['item_ids'].shape)
    print(item_batch_inp['item_batch_feature'].shape)

    item_all_emb, user_gcn_emb = model.new_generate(item_batch_inp)

    fw = open(out_dir, 'w')
    fw.write('user item score\n')
    for u_batch_id in tqdm(range(n_user_batchs)):
        start = u_batch_id * u_batch_size
        end = min((u_batch_id + 1) * u_batch_size, n_test_users)

        user_list_batch = all_users[start: end]
        user_batch = torch.LongTensor(np.array(user_list_batch)).to(device)
        u_g_embeddings = user_gcn_emb[user_batch]

        if batch_test_flag:
            # batch-item test
            n_item_batchs = n_items // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), n_items))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, n_items)

                item_batch = torch.LongTensor(np.array(range(i_start, i_end))).view(i_end-i_start).to(device)
                i_g_embddings = item_all_emb[item_batch]

                i_rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach().cpu()

                rate_batch[:, i_start: i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == n_items
        else:
            # all-item test
            item_batch = torch.LongTensor(np.array(range(0, n_items))).view(n_items, -1).to(device)
            i_g_embddings = item_all_emb[item_batch]
            rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach().cpu()

        user_batch_rating_uid = zip(rate_batch, user_list_batch)
        # print(user_list_batch)
        for u_id in user_list_batch:
            for i_id in range(len(item_all_emb)):
                fw.write('{} {} {}\n'.format(u_id, i_id,rate_batch[u_id % u_batch_size,i_id]))
                # print('{} {} {}\n'.format(u_id, i_id, rate_batch[u_id,i_id]))

    # assert count == n_users
    pool.close()
    return True 
    