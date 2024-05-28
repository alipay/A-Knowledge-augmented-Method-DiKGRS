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
import argparse

def generate_triplets(inverse , fname):
    can_triplets_np = np.loadtxt(fname, dtype=np.int32)
    can_triplets_np = np.unique(can_triplets_np, axis=0)

    if inverse:
        # get triplets with inverse direction like <entity, is-aspect-of, item>
        inv_triplets_np = can_triplets_np.copy()
        inv_triplets_np[:, 0] = can_triplets_np[:, 2]
        inv_triplets_np[:, 2] = can_triplets_np[:, 0]
        inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1
        # get full version of knowledge G
        triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)
    else:
        # consider two additional relations --- 'interact'.
        triplets = can_triplets_np.copy()
    return triplets
    
def build_G(triplets):
    ckg_G = nx.MultiDiGraph()
    rd = defaultdict(list)
    print("\nBegin to load knowledge G triples ...")
    for h_id, r_id, t_id in tqdm(triplets, ascii=True):
        ckg_G.add_edge(h_id, t_id, relationship=r_id)
        rd[r_id].append([h_id, t_id])

    return ckg_G, rd
def extract_type(relation_fname):
    head_type = {}
    tail_type = {}
    relation_type = {}
    with open(relation_fname) as f:
        for line in f:
            if line.startswith("http://rdf.freebase.com/ns/"):
                parts = line.strip().split()
                context = parts[0]
                rel_id = parts[1]
                relation_type[int(rel_id)] = context
                first_word = context.split('/')[-1].split('.')[0]
                last_word = context.split('/')[-1].split('.')[-1]
                head_type[int(rel_id)] = first_word
                tail_type[int(rel_id)] = last_word
            if line.startswith('http://www.w3.org/'):
                parts = line.strip().split()
                context = parts[0]
                rel_id = parts[1]
                relation_type[int(rel_id)] = context
                first_word = context.split('/')[-1].split('#')[0]
                last_word = context.split('/')[-1].split('#')[-1]
                head_type[int(rel_id)] = first_word
                tail_type[int(rel_id)] = last_word
            #in movie dataset
            else:
                if 'org_id' not in line:
                    parts = line.strip().split()
                    context = parts[0]
                    rel_id = parts[1]
                    relation_type[int(rel_id)] = context
                    first_word = context.split('.')[0]
                    last_word = context.split('.')[-1]
                    head_type[int(rel_id)] = first_word
                    tail_type[int(rel_id)] = last_word
                    
                
    return head_type,tail_type,relation_type
    
def extract_entity_type(G,file_path, head_type, tail_type):
    node_type_dict = {}
    hash_type_dict = {}
    hash_val = 1 
    num_entity = G.number_of_nodes()
    with open(file_path, 'w') as file:
        file.write('entity_id type\n')
        for node in tqdm(range(num_entity)):
            out_edge_types = [k for u, v,k in G.out_edges(node,data = 'relationship')]
            in_edge_types = [k for u, v, k in G.in_edges(node, data = 'relationship')]
            
            head_list = [head_type[item] for item in out_edge_types if item in head_type.keys() ]
            tail_list = [tail_type[item] for item in in_edge_types if item in tail_type.keys()  ]
            type_list = head_list + tail_list
            # print(node,type_list)
            counts = Counter(type_list)
            
            if len([x for x in counts  if x != 'type' and x != 'instance']) != 0:
                type = max([x for x in counts  if x != 'type' and x != 'instance'], key=counts.get)
            else:
                type = max([x for x in counts], key=counts.get)
            
            node_type_dict[node] = type
            if type not in hash_type_dict:
                hash_type_dict[type] = hash_val
                hash_val += 1
            file.write('{} {}\n'.format(node,hash_type_dict[type]))
    return hash_type_dict

        
def get_all_metapath(node,G):
    paths = []
    out_path = G.out_edges(node,data = 'relationship')
    in_path = G.in_edges(node,data = 'relationship')
    paths = set(list(out_path) + list(in_path))
    # extract mapping between entity and type
    meta_path = [ [item[0],item[2],item[1]] for item in paths]
    return meta_path

def generate_WL_kernel_similarity(G_filtered,node1,node2):
    subG1 = nx.ego_graph(G_filtered, node1, radius=2)
    print(len(subG1.nodes()))
    subG2 = nx.ego_graph(G_filtered, node2, radius=2)
    # print(subG1.nodes())
    adj_matrix_1 = nx.to_numpy_matrix(subG1)
    adj_matrix_2 = nx.to_numpy_matrix(subG2)
    node_relabel1 = {}
    for i,node in enumerate(subG1.nodes()):
        if list(subG1.nodes())[i] == node1:
            node_relabel1[i] = node1
        else:
            node_relabel1[i] = list(subG1.nodes())[i]
    node_relabel2 = {}
    for i in range(len(subG2.nodes())):
        if list(subG2.nodes())[i] == node2:
            node_relabel2[i] = node2
        else:
            node_relabel2[i] = list(subG2.nodes())[i]
    # print(node_relabel2)
    # print(adj_matrix_1.shape,node_relabel1)
    G1 = G(np.array(adj_matrix_1),node_labels = node_relabel1)
    G2 = G(np.array(adj_matrix_2), node_labels = node_relabel2)
    gk = WeisfeilerLehman(normalize = True)
    X = gk.fit_transform([G1,G2])
    return X[0,1]

def build_metapath_lookup_table(G,entity_lookup_table,metapath_fname):
    metapath_list = []
    for node in tqdm(range(G.number_of_nodes())):
        meta_paths = get_all_metapath(node,G)
        for path in meta_paths:
            meta_path = [entity_lookup_table[path[0]],path[1],entity_lookup_table[path[2]]]
            if meta_path not in metapath_list:
                metapath_list.append(meta_path)
    meta_docs = [['E'+str(line[0]),'V'+str(line[1]),'E'+str(line[2])] for line in metapath_list]
    tagged_data = [TaggedDocument(words = meta_docs[i], tags = 'doc'+str(i)) for i in range(len(meta_docs))]
    model = Doc2Vec(vector_size = 8, min_count=2, epochs=40)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    metapath_dict = {}
    for i in range(len(meta_docs)):
        metapath_dict[i] = model.infer_vector(meta_docs[i])
    with open(metapath_fname,'w') as f:
        f.write('head_type relation_type node_type representation\n')
        for r in range(len(metapath_list)):
            elements = ' '.join(str(element) for element in metapath_dict[r])
            f.write('{} {} {} {}\n'.format(metapath_list[r][0],metapath_list[r][1],metapath_list[r][2],elements))
    return

def generate_node_metapath_representation(G,node,metapath_list,metapath_dict,entity_lookup_table,pooling_type):
    paths = get_all_metapath(node,G)
    #turn entity id into type
    paths = [[entity_lookup_table[path[0]],path[1],entity_lookup_table[path[2]]] for path in paths]
    paths = np.unique(np.array(paths),axis = 0)
    paths_representation = [metapath_dict[metapath_list.index(list(path))] for path in paths]
    paths_representation = np.array(paths_representation)
    if pooling_type == 'mean':
        return np.mean(paths_representation,axis = 0)
    elif pooling_type == 'max':
        return np.max(paths_representation, axis = 0)

def neighbor_similarity(G_filtered,node1,node2,score_type,hop = 2):
    vec1, vec2 = np.zeros(hop * G_filtered.number_of_nodes()), np.zeros(hop * G_filtered.number_of_nodes())
    node_set_1 = [node1]
    node_set_2 = [node2]
    # default hop = 2
    for h in range(hop):
        for node_ego in node_set_1:
            for n in list(G_filtered.neighbors(node_ego)):
                vec1[n + h * G_filtered.number_of_nodes()] = 1
            node_set_1.remove(node_ego)
            node_set_1 = node_set_1 + list(G_filtered.neighbors(node_ego))

        for node_ego in node_set_2:
            for n in list(G_filtered.neighbors(node_ego)):
                vec2[n + h * G_filtered.number_of_nodes()] = 1
            node_set_2.remove(node_ego)
            node_set_2 = node_set_2 + list(G_filtered.neighbors(node_ego))

    if score_type == 'Jaccard':
        set1 = set([item for item in np.nonzero(vec1)[0]])
        set2 = set([item for item in np.nonzero(vec2)[0]])
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union
    elif score_type == 'Manhattan':
        return np.sum(np.abs(vec1 - vec2))
    elif score_type == 'cosine':
        return np.matmul(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    elif score_type == 'Hamming':
        return np.sum(vec1 != vec2)
    else:
        return 'unknown'
    
def check_similarity(G,G_filtered,node1,node2,entity_lookup_table,metapath_list,metapath_dict,score_type,pooling_type,alpha):
    # sim_score_k = generate_WL_kernel_similarity(G_filtered,node1,node2)
    sim_score_k = neighbor_similarity(G_filtered,node1,node2,score_type)
    node1_r = generate_node_metapath_representation(G,node1,metapath_list,metapath_dict,entity_lookup_table, pooling_type)
    # print(node1_r)
    node2_r = generate_node_metapath_representation(G,node2,metapath_list,metapath_dict,entity_lookup_table,pooling_type)
    # print(node2_r)
    sim_score_m = np.dot(node1_r,node2_r)/(np.linalg.norm(node1_r)*np.linalg.norm(node2_r))
    # print(sim_score_k,sim_score_m)
    return alpha*sim_score_k + (1-alpha) * sim_score_m


def read_entity_type_lookup_table(entity_type_fname):
    entity_lookup_table  = {}
    with open(entity_type_fname) as f:
        for line in f:
            if 'type' not in line:
                parts = line.strip().split()
                entity_lookup_table[int(parts[0])] = int(parts[1])
    return entity_lookup_table

def read_metapath_lookup_table(metapath_fname):
    metapath_list = []
    metapath_dict = {} 
    with open(metapath_fname) as f:
        for line in f:
            if 'head_type' not in line:
                parts = line.strip().split()
                metapath_list.append([int(parts[0]),int(parts[1]),int(parts[2])])
                metapath_dict[metapath_list.index([int(parts[0]),int(parts[1]),int(parts[2])])] = [float(item) for item in parts[3:] ]
    return metapath_list,metapath_dict

def filter_G(G,item_num):
    G_filtered = G.copy()
    for node in G.nodes():
        if len(G.out_edges(node)) > item_num:
            G_filtered.remove_node(node)
            print('Filtered :', node)
    return G_filtered


def node_check(node1,node2,G,G_filtered):
    subG1  = nx.ego_graph(G_filtered,node1,radius=2)
    subG2 = nx.ego_graph(G_filtered,node2,radius=2)
    print(subG1.nodes())
    print(subG2.nodes())
    plt.figure(figsize=(5, 5))
    nx.draw(subG1, with_labels=True)
    plt.title("SubG centered at node {}".format(node1))

    plt.figure(figsize=(5, 5))
    nx.draw(subG2, with_labels=True)
    plt.title("SubG centered at node {}".format(node2))

    metapath_1 = [[entity_lookup_table[path[0]],path[1],entity_lookup_table[path[2]]] for path in get_all_metapath(node1,G)] 
    metapath_2 =  [[entity_lookup_table[path[0]],path[1],entity_lookup_table[path[2]]] for path in get_all_metapath(node2,G)] 
    # print(len([item for item in metapath_1 if item in metapath_2]))

    node1_r = generate_node_metapath_representation(G,node1,metapath_list,metapath_dict,entity_lookup_table,'mean')
    node2_r = generate_node_metapath_representation(G,node2,metapath_list,metapath_dict,entity_lookup_table,'mean')
    # print(np.dot(node1_r,node2_r)/(np.linalg.norm(node1_r)*np.linalg.norm(node2_r)))
    plt.show()

    print(check_similarity(G,G_filtered,node1,node2,entity_lookup_table,metapath_list,metapath_dict,'mean',1))
    print(check_similarity(G,G_filtered,node1,node2,entity_lookup_table,metapath_list,metapath_dict,'mean',0))
    print(check_similarity(G,G_filtered,node1,node2,entity_lookup_table,metapath_list,metapath_dict,'mean',0.5))

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


def generate_pseudo_entries(rating_all_fname, path, dataset, score_type = 'Jaccard', pooling = 'mean', score_lambda = 0.5, sim_threshold = 0.85, file_exist = 1, abs_mode = 0):
    kg_fname =  path + dataset + '/' +'kg_final.txt'
    print('loading triplets ...')
    triplets = generate_triplets(1,fname = kg_fname)
    print('generating graph ...')
    G,rd = build_G(triplets)

    # filter G
    if dataset =='movie':
        item_num = 2347
    if dataset == 'last-fm':
        item_num = 3915	

    num_entity = G.number_of_nodes()
    if dataset != 'mybank1' and dataset != 'mybank2':
        G_filtered = filter_G(G,item_num)
    else:
        G_filtered = G
    

    # entity_list
    if dataset != 'mybank1' and dataset != 'mybank2':
        relation_list_fname = path + dataset + '/' + 'relation_list.txt'
        head_type, tail_type, relation_type = extract_type(relation_list_fname)
        file_path = path + dataset + '/' + 'entity_type_list.txt'
        hash_type_dict = extract_entity_type(G,file_path, head_type, tail_type)
    else:
        file_path = path + dataset + '/' + 'entity_type_list.txt'

    #build_metapath
    metapath_fname = path + dataset + '/' + 'metapath_list.txt'
    entity_lookup_table = read_entity_type_lookup_table(file_path)
    build_metapath_lookup_table(G,entity_lookup_table,metapath_fname)
    metapath_list, metapath_dict = read_metapath_lookup_table(metapath_fname)

    print('Pseudo rating generating...')
    if not abs_mode:
        w_fname = path + dataset + '/' + 'pseudo_ratings.txt'
    else:
        w_fname = path + dataset + '/' + 'pseudo_ratings_abs.txt'

    # build item_exp_count table
    item_exp_count = {}
    total_count = 0
    user_interactions = {}
    user_infos = {}
    with open(rating_all_fname) as f:
        for line in f:
            if 'user_id' not in line:
                total_count += 1
                parts = line.strip().split()
                if int(parts[1]) not in item_exp_count.keys():
                    item_exp_count[int(parts[1])] = 1
                else:
                    item_exp_count[int(parts[1])] += 1

                user_id = int(parts[0])
                if user_id not in user_interactions.keys():
                    user_interactions[user_id] = [(int(parts[1]),int(parts[2]))]
                    if dataset == 'movie':
                        user_infos[user_id] = [int(parts[3]),int(parts[4]),int(parts[5]),int(parts[6])]
                else:
                    user_interactions[user_id] = user_interactions[user_id] + [(int(parts[1]),int(parts[2]))]
    num_users = max(user_interactions.keys())
    print(num_users)
    #get top k percentage pv items
    num_items = max(list(item_exp_count.keys()))
    print(num_items)
    sorted_item_exp = {k: v for k, v in sorted(item_exp_count.items(), key=lambda item: item[1])}
    min_score = np.inf
    max_score = 0

    if file_exist:
        if dataset == 'movie' and abs_mode == 0:
            with open('similarity_lookup.pickle', 'rb') as f:
                lookup_t = pickle.load(f)
        elif dataset == 'movie' and abs_mode == 1:
            with open('similarity_lookup_abs.pickle', 'rb') as f:
                lookup_t = pickle.load(f)
        elif dataset =='last-fm':
            with open('last-fm_similarity_lookup.pickle', 'rb') as f:
                lookup_t = pickle.load(f)
        elif dataset == 'mybank1':
            with open('mybank1_similarity_lookup.pickle', 'rb') as f:
                lookup_t = pickle.load(f)
        elif dataset == 'mybank2':
            with open('mybank2_similarity_lookup.pickle', 'rb') as f:
                lookup_t = pickle.load(f)
    else:
        print('generating lookup table...')
        lookup_t = np.zeros((num_items,num_items))
        for item_i in tqdm(range(num_items)):
            for item_j in range(item_i+1,num_items):
                lookup_t[item_i][item_j] = check_similarity(G,G_filtered,item_i,item_j,entity_lookup_table,metapath_list, metapath_dict, score_type, pooling, score_lambda)
                lookup_t[item_j][item_i] = lookup_t[item_i][item_j]
                if dataset == 'mybank1' or dataset == 'mybank2':
                    if lookup_t[item_i][item_j] < min_score:
                        min_score = lookup_t[item_i][item_j].copy()
                else:
                    if lookup_t[item_i][item_j] > max_score:
                        max_score = lookup_t[item_i][item_j].copy()

        if dataset=='movie' and not abs_mode:
            with open('similarity_lookup.pickle', 'wb') as f:
                pickle.dump(lookup_t, f)
        elif dataset=='movie' and abs_mode:
            with open('similarity_lookup_abs.pickle', 'wb') as f:
                pickle.dump(lookup_t, f)
        elif dataset == 'last-fm':
            with open('last-fm_similarity_lookup.pickle', 'wb') as f:
                pickle.dump(lookup_t, f)
        elif dataset == 'mybank1':
            with open('mybank1_similarity_lookup.pickle', 'wb') as f:
                pickle.dump(lookup_t, f)
        elif dataset == 'mybank2':
            with open('mybank2_similarity_lookup.pickle', 'wb') as f:
                pickle.dump(lookup_t, f)


    sim_item_dict = defaultdict(list)

    if dataset == 'mybank1' or dataset == 'mybank2':
        for item in range(num_items):
            sim_item_dict[item] = list(np.argwhere(lookup_t[item] <= sim_threshold).reshape(-1))
    else:
        for item in range(num_items):
            sim_item_dict[item] = list(np.argwhere(lookup_t[item] >= sim_threshold).reshape(-1))
    
    print(w_fname)

    with open(w_fname,'w') as fw:
        for user in tqdm(list(user_interactions.keys()), ascii=True):
            interacted_items = set([t[0] for t in user_interactions[user]])
            interacted_ratings = {t[0]:t[1] for t in user_interactions[user]}
            # top_k_items = set(list(item_under_review.keys()))
            for item_j in interacted_items:
                sim_items = sim_item_dict[item_j]
                for item_i in sim_items:
                    if item_i not in interacted_items:
                        if dataset == 'movie':
                            fw.write('{} {} {} {} {} {} {}\n'.format(user, item_i, interacted_ratings[item_j],user_infos[user][0], user_infos[user][1], user_infos[user][2], user_infos[user][3]))
                        else:
                            fw.write('{} {} {}\n'.format(user, item_i, interacted_ratings[item_j]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generating Pseudo Entries')
    parser.add_argument('--dataset', type=str, default='movie', help='choose dataset from [movie, last-fm, mybank1, mybank2]')
    parser.add_argument('--pooling', type=str, default='mean', help='choose pooling function from [mean, max]')
    parser.add_argument('--score_weight', type=float, default=0.5, help='the weight to balance neighborhood similarity and metapath similarity')
    parser.add_argument('--sim_threshold', type=float, default=0.95, help='the threshold value for filtering pseudo samples')
    parser.add_argument('--file_exist', type=bool, default= False, help='if the i2i similarity lookup table exists or not')
    args_config = parser.parse_args()

    path = '../data/'
    rating_all_fname = path + args_config.dataset + '/rating_all.txt'
    dataset = args_config.dataset
    if dataset == 'movie' or dataset == 'last-fm':
        score_type = 'Jaccard'
    else:
        score_type = 'Hamming'
    generate_pseudo_entries(rating_all_fname, path, dataset,  score_type = score_type, pooling = args_config.pooling, score_lambda = args_config.score_weight, sim_threshold = args_config.sim_threshold , file_exist= args_config.file_exist)

