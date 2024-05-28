import random
import torch
import argparse
import numpy as np
from utils.preprocess import generate_movie_rating_all_file
from modules.pseudo_entry_generation import generate_pseudo_entries
from time import time
from prettytable import PrettyTable
import pickle
from utils.parser import parse_args
from utils.data_loader import load_data
from torch.utils.data import DataLoader
from collections import defaultdict

from modules.VRKG_DVN import Recommender as VRKG_Recommender

from utils.evaluate import test
from utils.helper import early_stopping
import ast
import torch.nn.functional as F
from utils.evaluate import score 
from utils.data_loader import get_train_test_data
from modules.KGIN_DVN import Recommender as KGIN_Recommender




n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0


def get_feed_dict(train_entity_pairs, start, end, train_user_set):
    def negative_sampling(user_item, train_user_set):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            while True:
                neg_item = np.random.randint(low=0, high=n_items, size=1)[0]
                if neg_item not in train_user_set[user]:
                    break
            neg_items.append(neg_item)
        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end].to(device)
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(negative_sampling(entity_pairs, train_user_set)).to(device)
    return feed_dict


def get_kg_dict(train_kg_pairs, start, end, relation_dict):
    def negative_sampling(kg_pairs, relation_dict):
        neg_ts = []
        for h, r, _ in kg_pairs.cpu().numpy():
            r = int(r)
            h = int(h)
            while True:
                neg_t = np.random.randint(low=0, high=n_entities, size=1)[0]
                if (h, neg_t) not in relation_dict[r]:
                    break
            neg_ts.append(neg_t)
        return neg_ts

    kg_dict = {}
    kg_pairs = train_kg_pairs[start:end].to(device)
    kg_dict['h'] = kg_pairs[:, 0]
    kg_dict['r'] = kg_pairs[:, 1]
    kg_dict['pos_t'] = kg_pairs[:, 2]
    kg_dict['neg_t'] = torch.LongTensor(negative_sampling(kg_pairs, relation_dict)).to(device)
    return kg_dict



if __name__ == '__main__':
    """fix the random seed"""
    global args, device
    args = parse_args()
    device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")
    print(device)

    seed = args.random_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    rating_all_fname = args.data_path + args.dataset + '/rating_all.txt'
    path = '/ossfs/workspace/data/'
    ps_fname = args.data_path + args.dataset + '/pseudo_ratings.txt'
    if args.abs_ps_a or args.abs_ps_b:
        dataset = 'movie'
        score_lambda = int(args.abs_ps_b)
        if args.abs_ps_a:
            sim_threshold = 0.999
            file_exist = False
        elif args.abs_ps_b:
            sim_threshold = 0.701
            file_exist = True
        max_score = generate_pseudo_entries(rating_all_fname, path, dataset, top_p = 0.007, score_type = 'Jaccard', pooling = 'mean', score_lambda = score_lambda, sim_threshold = sim_threshold, file_exist = file_exist, abs_mode = 1 )
        ps_fname = args.data_path + args.dataset + '/pseudo_ratings_abs.txt'
    

    """build dataset"""
    get_train_test_data(rating_all_fname, ps_fname, args.data_path + args.dataset +'/test_data.pkl', path = args.data_path, num_ps = args.num_ps, dataset=args.dataset, backbone = args.backbone)
    train_cf, test_cf, user_dict, n_params, graph, triplets, relation_dict, mat_list, pseudo_flag_dict, item_feature_dict = load_data(args)
    adj_mat_list, norm_mat_list, mean_mat_list = mat_list


    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']
    

    """cf data"""
    train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in test_cf], np.int32))


    n_params['dvn_input_dim'] = len(item_feature_dict[0])

    # print(item_feature_dict)
    """kg data"""
    train_kg_pairs = torch.LongTensor(np.array([[kg[0], kg[1], kg[2]] for kg in triplets], np.int32))

    """define model"""
    if args.backbone == 'VRKG':
        model = VRKG_Recommender(n_params, args, graph, mean_mat_list[0]).to(device)
    elif args.backbone =='KGIN':
        model = KGIN_Recommender(n_params, args, graph, mean_mat_list[0]).to(device)
    else:
        print('Model Unsupported!')

    print(model)
    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cur_best_pre_0 = 0
    stopping_step = 0
    best_epoch = 0
    should_stop = False

    if args.score_mode == True:
        print('scoring...')
        model.load_state_dict(torch.load('/ossfs/workspace/KGIN_movie_4k_64_movie.ckpt'))
        model.eval()
        ret = score(model, user_dict, n_params, item_feature_dict, 'KGIN_movie_4k_ours.txt')

    if args.score_mode == False:
        write_fname =  'Logs/' + args.log_fname
        fw = open(write_fname,'w')
        fg = open('gate_record.txt','w')
        # print(args.add_dvn)
        fw.write(str(args.dvn_dropout_rate)+'\n'  + str(args.add_dvn) + '\n' +
                str(args.add_pseudo) + '\n' + 
                str(args.pseudo_weight) + '\n' + args.dvn_hidden_dim + '\n')

        print("start training ...")
        for epoch in range(args.epoch):
            index = np.arange(len(train_cf))
            np.random.shuffle(index)
            train_cf_pairs = train_cf_pairs[index]

            """ training """
            """ train cf """
            mf_loss_total, s = 0, 0
            loss, cor_loss = 0, 0
            train_cf_s = time()
            model.train()
            flag = 'cf'
            gate_epoch = []
            gate_num = 0
            ps_count = []
            while s + args.batch_size <= len(train_cf):
                cf_batch = get_feed_dict(train_cf_pairs,
                                        s, s + args.batch_size,
                                        user_dict['train_user_set'])
                cf_batch['dvn_pos_feature'] = torch.tensor(np.array([item_feature_dict[x.item()] for x in cf_batch['pos_items']])).float().to(device)#users torch.Size([1024])
                cf_batch['dvn_neg_feature'] = torch.tensor(np.array([item_feature_dict[x.item()] for x in cf_batch['neg_items']])).float().to(device)#pos_items torch.Size([1024])，neg_items torch.Size([1024])
                cf_batch['pseudo_label_pos'] = torch.tensor([pseudo_flag_dict[(cf_batch['users'][i].item(), cf_batch['pos_items'][i].item())] for i in range(len(cf_batch['pos_items']))]).long().to(device)#dvn_pos_feature torch.Size([1024, 78])
                """pseudo_label_neg always = 1"""
                cf_batch['pseudo_label_neg'] = torch.tensor([1 for i in range(len(cf_batch['neg_items']))]).long().to(device)#dvn_neg_feature torch.Size([1024, 78])

                
                ps_label = torch.add(-1, cf_batch['pseudo_label_pos'].reshape(-1))
                ps_count.append(-1 * torch.sum(ps_label, dim = -1))

                if args.backbone =='VRKG':
                    batch_loss, mf_loss, _  = model(cf_batch)
                    optimizer.zero_grad()
                    mf_loss.backward()
                    optimizer.step()

                    mf_loss_total += mf_loss.item()
                    s += args.batch_size
            
                if args.backbone =='KGIN':
                    batch_loss, _, _, batch_cor = model(cf_batch)
                    batch_loss = batch_loss
                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()

                    loss += batch_loss
                    cor_loss += batch_cor
                    s += args.batch_size
            

                
                    
            train_cf_e = time()
            if epoch % 10 == 9 or epoch == 0:
                """testing"""
                test_s_t = time()
                model.eval()
                ret = test(model, user_dict, n_params, item_feature_dict)

                test_e_t = time()

                train_res = PrettyTable()
                train_res.field_names = ["Epoch", "training time", "testing time", "recall", "ndcg", "precision",
                                        "hit_ratio", "auc", "f1"]
                train_res.add_row(
                    [epoch, train_cf_e - train_cf_s, test_e_t - test_s_t, ret['recall'], ret['ndcg'], ret['precision'],
                    ret['hit_ratio'], ret['auc'], ret['f1']]
                )
                print(train_res)
                fw.write(str(train_res))
                fw.write('\n')

                # *********************************************************
                # early stopping when cur_best_pre_0 is decreasing for 10 successive steps.
                cur_best_pre_0, stopping_step, should_stop, best_epoch = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                                        stopping_step, best_epoch, epoch,
                                                                                        expected_order='acc',
                                                                                        flag_step=10)
                if should_stop:
                    break

                """save weight"""
                if ret['recall'][0] == cur_best_pre_0 and args.save:
                    torch.save(model.state_dict(), args.log_fname[:-4]+ '_' + args.dataset + '.ckpt')
            
                if args.backbone =='VRKG':
                    print('using time %.4f, training loss at epoch %d: %.4f' % (train_cf_e - train_cf_s, epoch, mf_loss_total))
                    # print(gate_epoch)
                    fw.write('using time %.4f, training loss at epoch %d: %.4f\n' % (train_cf_e - train_cf_s, epoch, mf_loss_total))
                if args.backbone == 'KGIN':
                    print('using time %.4f, training loss at epoch %d: %.4f, cor: %.6f' % (train_cf_e - train_cf_s, epoch, loss.item(), cor_loss.item()))
                    fw.write('using time %.4f, training loss at epoch %d: %.4f, cor: %.6f\n' % (train_cf_e - train_cf_s, epoch, loss.item(), cor_loss.item()))

        # fg.write(' '.join(str(x) for x in gate_epoch.items()))
        print('stopping at %d, recall@20:%.4f' % (epoch, ret['recall'][0]))
        fw.write('stopping at %d, recall@20:%.4f\n' % (epoch, ret['recall'][0]))
        print('the best epoch is at %d, recall@20:%.4f' % (best_epoch, cur_best_pre_0))
        fw.write('the best epoch is at %d, recall@20:%.4f\n' % (best_epoch, cur_best_pre_0))
        fw.close()







