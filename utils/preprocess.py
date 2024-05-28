import networkx as nx
from tqdm import tqdm
from collections import defaultdict
# from grakel import Graph
# from grakel.kernels import WeisfeilerLehman
import networkx as nx
from tqdm import tqdm
from collections import defaultdict
from collections import Counter
import networkx as nx
from tqdm import tqdm
from collections import defaultdict
from collections import Counter
import numpy as np
# from grakel.kernels import GraphletSampling
# from grakel.kernels import RandomWalk
# from grakel.kernels import WeisfeilerLehman
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import pickle


def zipcode_encode(zipcode):
    # 以州的首字母缩写为键，对应的邮政编码范围为值
    zip_state_mapping = {
        "AL": ("35004", "36925"), "AK": ("99501", "99950"), "AZ": (	"85001", "86556"), 
        "AR": ("71601" , "72959"), "CA": ("90001", "96162"), "CO": ("80001" , "81658"), 
        "CT": ("06001", "06928"), "DE": ("19701" , "19980"), "FL": ("32003" , "34997"), 
        "GA": ("30002", "39901"), "HI": ("96701" , "96898"), "ID": ("83201" , "83877"), 
        "IL": ("60001" , "62999"), "IN": ("46001" , "47997"), "IA": ("50001" , "52809"), 
        "KS": ("66002" , "67954"), "KY": ("40003" , "42788"), "LA": ("70001" , "71497"), 
        "ME": ("03901" , "04992"), "MD": ("20588" , "21930"), "MA": ("01001" , "05544"), 
        "MI": ("48001" , "49971"), "MN": ("55001" , "56763"), "MS": ("38601" , "39776"), 
        "MO": ("63001" , "65899"), "MT": ("59001" , "59937"), "NE": ("68001" , "69367"), 
        "NV": ("88901" , "89883"), "NH": ("03031" , "03897"), "NJ": ("07001" , "08989"), 
        "NM": ("87001" , "88439"), "NY": ("00501" , "14925"), "NC": ("27006" , "28909"),
        "ND": ("58001" , "58856"), "OH": ("43001" , "45999"), "OK": ("73001" , "74966"), 
        "OR": ("97001" , "97920"), "PA": ("15001" , "19640"), "RI": ("02801" , "02940"), 
        "SC": ("29001" , "29945"), "SD": ("57001" , "57799"), "TN": ("37010" , "38589"), 
        "TX": ("73301" , "88595"), "UT": ("84001" , "84791"), "VT": ("05001" , "05907"), 
        "VA": ("20101" , "24658"), "WA": ("98001" , "99403"), "WV": ("24701" , "26886"), 
        "WI": ("53001" , "54990"), "WY": ("82001" , "83414"), "GM":("96910", "96932"),
        'DC':("20000","20373")
    }

    # 去除邮政编码中的非数字字符
    zipcode = ''.join(filter(str.isdigit, zipcode))

    # 检查输入的邮政编码长度是否为5
    
    # 取前两位或前三位进行匹配
    zip_prefix = int(zipcode[:5])
    # 遍历映射字典，查找对应的州
    for state, (start, end) in zip_state_mapping.items():
        if int(start) <= zip_prefix <= int(end):
            return list(zip_state_mapping.keys()).index(state)

    
    return len(list(zip_state_mapping.keys()))+1 # 如果邮政编码无法找到对应的州，则返回 Unknown 或者其他适当的默认值

def load_rating(args):
    print('reading training file and testing file ...')
    directory = 'data/' + args.dataset
    # rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int64)
    with open(directory + "/train_data.pkl", 'rb') as fo:
        train_data = pickle.load(fo, encoding='bytes')
    with open(directory + "/test_data.pkl", 'rb') as fi:
        test_data = pickle.load(fi, encoding='bytes')
    rating_np = np.concatenate((train_data, test_data), axis=0)

    # reading rating file
    n_user = max(set(rating_np[:, 0])) + 1  # the result = max(rating_np[:, 0]
    n_item = max(set(rating_np[:, 1])) + 1  # the result = max(rating_np[:, 1]

    return n_user, n_item, train_data, test_data

def item_pv_table(rating_all_fname):
    item_exp_count = {}
    total_count = 0
    with open(rating_all_fname) as f:
        for line in f:
            if 'user_id' not in line:
                total_count += 1
                parts = line.strip().split()
                if int(parts[1]) not in item_exp_count.keys():
                    item_exp_count[int(parts[1])] = 1
                else:
                    item_exp_count[int(parts[1])] += 1
    return item_exp_count

def generate_movie_rating_all_file(data_path, dataset):
    if dataset == 'movie':
        path = data_path
        hash_table_fname = path + dataset + '/' + 'item_index2entity_id.txt'
        hash_dict = {}

        with open(hash_table_fname,'r') as f:
            for line in f:
                parts = line.strip().split()
                # print(parts)
                hash_dict[int(parts[0])] = int(parts[1])
                
        users_fname = path + 'movie'  + '/' + 'users.txt'
        user_dict = {}
        with open(users_fname,'r') as f:
            for line in f:
                parts = line.strip().split('::')
                user_dict[int(parts[0])] = parts[1:]
        
        with open(path + 'movie-VRKG4Rec' + "/test_data.pkl", 'rb') as fi:
            test_data = pickle.load(fi, encoding='bytes')
        
        ratings_fname = path + 'movie'  + '/' + 'ratings.txt'
        user_pos_ratings = dict()
        user_neg_ratings = dict()
        for line in open(ratings_fname, encoding='utf-8').readlines():
                array = line.strip().split("::")
                item_index_old = int(array[1])
                if item_index_old not in hash_dict:  # the item is not in the final item set
                    continue
                item_index = hash_dict[item_index_old]

                user_index_old = int(array[0])

                rating = float(array[2])
                if rating >= 4:
                    if user_index_old not in user_pos_ratings:
                        user_pos_ratings[user_index_old] = set()
                    user_pos_ratings[user_index_old].add(item_index)
                '''降低neg sample 阀值'''
                if rating <= 2:
                    if user_index_old not in user_neg_ratings:
                        user_neg_ratings[user_index_old] = set()
                    user_neg_ratings[user_index_old].add(item_index)


        rating_all_fname = path + 'movie'  + '/' + 'rating_all.txt'
        uid_hash = {}
        uid_value = 0
        with open(rating_all_fname,'w') as fw:
            for userid, pos_itemset in tqdm(user_pos_ratings.items()):
                user_infos = user_dict[userid]
                user_gender = int(user_infos[0] == 'M')
                zipcode = zipcode_encode(user_infos[3])
                if userid not in uid_hash:
                    uid_hash[userid] = uid_value
                    uid_value += 1
                for item in pos_itemset:
                    if np.any(np.all(test_data == [uid_hash[userid], item, 1], axis=1)) == False:
                        fw.write('{} {} {} {} {} {} {}\n'.format(uid_hash[userid], item, 1, user_gender, int(user_infos[1]), int(user_infos[2]), zipcode))
            for userid, neg_itemset in user_neg_ratings.items():
                if userid in uid_hash:
                    user_infos = user_dict[userid]
                    user_gender = int(user_infos[0] == 'M')
                    zipcode = zipcode_encode(user_infos[3])
                    for item in neg_itemset:
                        fw.write('{} {} {} {} {} {} {}\n'.format(uid_hash[userid], item, 0, user_gender, int(user_infos[1]), int(user_infos[2]), zipcode))
        print("Movie rating_all.txt file has generated")
        
    if dataset == 'music':
        user_pos_ratings = defaultdict(list)
        user_neg_ratings = defaultdict(list)
        with open(data_path + 'last-fm' + '/ratings_final.txt', 'r') as f:  
            for line in f:
                parts = line.strip().split()
                user = int(parts[0])
                item = int(parts[1])
                rating = int(parts[2])
                if rating == 1:
                    user_pos_ratings[user].append(item)
                else:
                    user_neg_ratings[user].append(item)
        
        with open(data_path +'last-fm' + "/test_data.pkl", 'rb') as fi:
            test_data = pickle.load(fi, encoding='bytes')
        
                
        rating_all_fname = data_path + 'last-fm'  + '/' + 'rating_all.txt'
        with open(rating_all_fname,'w') as fw:
            for userid, pos_itemset in tqdm(user_pos_ratings.items()):
                for item in pos_itemset:
                    if np.any(np.all(test_data == [userid, item, 1], axis=1)) == False:
                        fw.write('{} {} {}\n'.format(userid, item, 1))
            for userid, neg_itemset in user_neg_ratings.items():
                for item in neg_itemset:
                    fw.write('{} {} {}\n'.format(userid, item, 0))
        print("Last-fm rating_all.txt file has generated")


    return rating_all_fname


def stats_onhot(rating_all_fname):
    exp_dict = dict()
    pos_dict = dict()
    ratings_all = np.loadtxt(ratings_all_fname, dtype = np.int32)
    for line  in range(len(ratings_all[:,1])):
        item = ratings_all[line,1]
        rating = ratings_all[line,2]
        if item not in exp_dict:
            exp_dict[item] = 1
            pos_dict[item] = rating
        else:
            exp_dict[item] += 1
            pos_dict[item] += rating

    ctr_dict = [pos_dict[item] / exp_dict[item] for item in exp_dict]

if __name__ == '__main__':
    path = '/ossfs/workspace/data/'
    rating_all_fname = generate_movie_rating_all_file(path,'music')