import torch
import torch.nn as nn
import torch.nn.functional as F


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)

class SimpleFusion(nn.Module):
    def __init__(self,embedding_dim,dropout_rate):
        super(SimpleFusion, self).__init__()
        self.embedding_dim = embedding_dim
        self.MapLayer = nn.Linear(embedding_dim * 2, embedding_dim)
        self.dropout = nn.Dropout(p = dropout_rate )
    def forward(self,input1, input2):
        concatenated = torch.squeeze(torch.cat((input1, input2), dim=-1))
        inp = self.dropout(concatenated)
        output = self.MapLayer(inp)
        return output


class Aggregator(nn.Module):

    def __init__(self, in_dim, out_dim, dropout, aggregator_type):
        super(Aggregator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type

        self.message_dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()

        if self.aggregator_type == 'gcn':
            self.linear = nn.Linear(self.in_dim, self.out_dim)       # W in Equation (6)
            nn.init.xavier_uniform_(self.linear.weight)

        elif self.aggregator_type == 'graphsage':
            self.linear = nn.Linear(self.in_dim * 2, self.out_dim)   # W in Equation (7)
            nn.init.xavier_uniform_(self.linear.weight)

        elif self.aggregator_type == 'bi-interaction':
            self.linear1 = nn.Linear(self.in_dim, self.out_dim)      # W1 in Equation (8)
            self.linear2 = nn.Linear(self.in_dim, self.out_dim)      # W2 in Equation (8)
            nn.init.xavier_uniform_(self.linear1.weight)
            nn.init.xavier_uniform_(self.linear2.weight)

        else:
            raise NotImplementedError


    def forward(self, ego_embeddings, A_in):
        """
        ego_embeddings:  (n_users + n_entities, in_dim)
        A_in:            (n_users + n_entities, n_users + n_entities), torch.sparse.FloatTensor
        """
        # Equation (3)
        side_embeddings = torch.matmul(A_in, ego_embeddings)

        if self.aggregator_type == 'gcn':
            # Equation (6) & (9)
            embeddings = ego_embeddings + side_embeddings
            embeddings = self.activation(self.linear(embeddings))

        elif self.aggregator_type == 'graphsage':
            # Equation (7) & (9)
            embeddings = torch.cat([ego_embeddings, side_embeddings], dim=1)
            embeddings = self.activation(self.linear(embeddings))

        elif self.aggregator_type == 'bi-interaction':
            # Equation (8) & (9)
            sum_embeddings = self.activation(self.linear1(ego_embeddings + side_embeddings))
            bi_embeddings = self.activation(self.linear2(ego_embeddings * side_embeddings))
            embeddings = bi_embeddings + sum_embeddings

        embeddings = self.message_dropout(embeddings)           # (n_users + n_entities, out_dim)
        return embeddings


class KGAT(nn.Module):

    def __init__(self, args,
                 n_users, n_entities, n_relations, A_in=None,
                 user_pre_embed=None, item_pre_embed=None):

        super(KGAT, self).__init__()
        self.use_pretrain = args.use_pretrain

        self.n_users = n_users
        self.n_entities = n_entities
        self.n_relations = n_relations

        self.embed_dim = args.embed_dim
        self.relation_dim = args.relation_dim

        self.aggregation_type = args.aggregation_type
        self.conv_dim_list = [args.embed_dim] + eval(args.conv_dim_list)
        self.mess_dropout = eval(args.mess_dropout)
        self.n_layers = len(eval(args.conv_dim_list))

        self.kg_l2loss_lambda = args.kg_l2loss_lambda
        self.cf_l2loss_lambda = args.cf_l2loss_lambda

        self.entity_user_embed = nn.Embedding(self.n_entities + self.n_users, self.embed_dim)
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)
        self.trans_M = nn.Parameter(torch.Tensor(self.n_relations, self.embed_dim, self.relation_dim))

        
        self.add_dvn = args.add_dvn
        self.abs_dvn_fusion = args.abs_dvn_fusion
        self.abs_dvn_up = args.abs_dvn_up
        self.abs_dvn_pop = args.abs_dvn_pop

        
        if args.add_dvn and not args.abs_dvn_fusion:
            self.dvn_dropout_rate = args.dvn_dropout_rate
            self.dvn_dropout = nn.Dropout(p = self.dvn_dropout_rate )
            if args.dataset =='movie' and not args.abs_dvn_up:
                dvn_input_dim = 32
                self.dvn = nn.Sequential(
                    nn.Linear(dvn_input_dim,self.embed_dim//2),
                    nn.ReLU(),
                    nn.Linear(self.embed_dim//2, 1),
                    nn.Sigmoid())
            
            if args.dataset =='movie' and args.abs_dvn_pop:
                dvn_input_dim = 30
                self.dvn = nn.Sequential(
                    nn.Linear(dvn_input_dim,self.embed_dim//2),
                    nn.ReLU(),
                    nn.Linear(self.embed_dim//2, 1),
                    nn.Sigmoid())
        
            if args.dataset =='last-fm' or args.abs_dvn_up:
                dvn_input_dim = 2
                self.dvn = nn.Sequential(
                    nn.Linear(dvn_input_dim,1),
                    nn.Sigmoid())

            if args.dataset =='mybank1':
                dvn_input_dim = 717
                self.dvn = nn.Sequential(
                    nn.Linear(dvn_input_dim,self.embed_dim),
                    nn.ReLU(),
                    nn.Linear(self.embed_dim, 1),
                    nn.Sigmoid())
            
            if args.dataset =='mybank2':
                dvn_input_dim = 710
                self.dvn = nn.Sequential(
                    nn.Linear(dvn_input_dim,self.embed_dim),
                    nn.ReLU(),
                    nn.Linear(self.embed_dim, 1),
                    nn.Sigmoid())
        
        if args.add_dvn and args.abs_dvn_fusion:
            self.dvn_dropout_rate = args.dvn_dropout_rate
            self.dvn_dropout = nn.Dropout(p = self.dvn_dropout_rate )
            self.emb_size = sum(self.conv_dim_list)
            self.dvn = nn.Sequential(nn.Linear(32, self.emb_size//2),
                                    nn.ReLU(),
                                    nn.Linear(self.emb_size//2, self.emb_size))
            self.SimpleFusion = SimpleFusion(embedding_dim = self.emb_size, dropout_rate = self.dvn_dropout_rate)


        if (self.use_pretrain == 1) and (user_pre_embed is not None) and (item_pre_embed is not None):
            other_entity_embed = nn.Parameter(torch.Tensor(self.n_entities - item_pre_embed.shape[0], self.embed_dim))
            nn.init.xavier_uniform_(other_entity_embed)
            entity_user_embed = torch.cat([item_pre_embed, other_entity_embed, user_pre_embed], dim=0)
            self.entity_user_embed.weight = nn.Parameter(entity_user_embed)
        else:
            nn.init.xavier_uniform_(self.entity_user_embed.weight)

        nn.init.xavier_uniform_(self.relation_embed.weight)
        nn.init.xavier_uniform_(self.trans_M)

        self.aggregator_layers = nn.ModuleList()
        for k in range(self.n_layers):
            self.aggregator_layers.append(Aggregator(self.conv_dim_list[k], self.conv_dim_list[k + 1], self.mess_dropout[k], self.aggregation_type))

        self.A_in = nn.Parameter(torch.sparse.FloatTensor(self.n_users + self.n_entities, self.n_users + self.n_entities))
        if A_in is not None:
            self.A_in.data = A_in
        self.A_in.requires_grad = False



    def calc_cf_embeddings(self):
        ego_embed = self.entity_user_embed.weight
        all_embed = [ego_embed]

        for idx, layer in enumerate(self.aggregator_layers):
            ego_embed = layer(ego_embed, self.A_in)
            norm_embed = F.normalize(ego_embed, p=2, dim=1)
            all_embed.append(norm_embed)

        # Equation (11)
        all_embed = torch.cat(all_embed, dim=1)         # (n_users + n_entities, concat_dim)
        return all_embed


    def calc_cf_loss(self, user_ids, item_pos_ids, item_pos_flag, item_pos_fea, item_neg_ids, item_neg_fea):
        """
        user_ids:       (cf_batch_size)
        item_pos_ids:   (cf_batch_size)
        item_neg_ids:   (cf_batch_size)
        """
        all_embed = self.calc_cf_embeddings()                       # (n_users + n_entities, concat_dim)
        user_embed = all_embed[user_ids]                            # (cf_batch_size, concat_dim)
        item_pos_embed = all_embed[item_pos_ids]                    # (cf_batch_size, concat_dim)
        item_neg_embed = all_embed[item_neg_ids]                    # (cf_batch_size, concat_dim)

        # Equation (12)
        pos_score = torch.sum(user_embed * item_pos_embed, dim=1)   # (cf_batch_size)
        neg_score = torch.sum(user_embed * item_neg_embed, dim=1)   # (cf_batch_size)

        if self.abs_dvn_up:
            item_pos_fea = item_pos_fea[:,:2]
            item_neg_fea = item_neg_fea[:,:2]
        if self.abs_dvn_pop:
            item_pos_fea = item_pos_fea[:,2:]
            item_neg_fea = item_neg_fea[:,2:]

        if self.add_dvn and not self.abs_dvn_fusion:
            item_pos_fea, item_neg_fea = self.dvn_dropout(item_pos_fea), self.dvn_dropout(item_neg_fea)
            dvn_pos_weight = self.dvn(item_pos_fea)
            cf_loss = F.logsigmoid(pos_score - neg_score)
            cf_loss_all_adj = dvn_pos_weight * torch.mul(torch.add(1, -1* item_pos_flag), cf_loss) + torch.mul(item_pos_flag, cf_loss)
            cf_loss = -1 * torch.mean(cf_loss_all_adj)
        
        elif  self.abs_dvn_fusion:
            item_pos_fea, item_pos_fea = self.dvn_dropout(item_pos_fea), self.dvn_dropout(item_pos_fea)
            dvn_pos_e, dvn_neg_e = self.dvn(item_pos_fea), self.dvn(item_pos_fea)
            item_pos_cf_embed, item_neg_cf_embed = self.SimpleFusion(dvn_pos_e,item_pos_embed), self.SimpleFusion(dvn_neg_e,item_neg_embed)
            pos_score = torch.sum(user_embed * item_pos_cf_embed, dim=1)    # (cf_batch_size)
            neg_score = torch.sum(user_embed * item_neg_cf_embed, dim=1)    # (cf_batch_size)
            cf_loss = (-1.0) * F.logsigmoid(pos_score - neg_score)
            cf_loss = torch.mean(cf_loss)
        else:
            cf_loss = (-1.0) * F.logsigmoid(pos_score - neg_score)
            cf_loss = torch.mean(cf_loss)
    
        l2_loss = _L2_loss_mean(user_embed) + _L2_loss_mean(item_pos_embed) + _L2_loss_mean(item_neg_embed)
        loss = cf_loss + self.cf_l2loss_lambda * l2_loss
        return loss


    def calc_kg_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.relation_embed(r)                                                # (kg_batch_size, relation_dim)
        W_r = self.trans_M[r]                                                           # (kg_batch_size, embed_dim, relation_dim)

        h_embed = self.entity_user_embed(h)                                             # (kg_batch_size, embed_dim)
        pos_t_embed = self.entity_user_embed(pos_t)                                     # (kg_batch_size, embed_dim)
        neg_t_embed = self.entity_user_embed(neg_t)                                     # (kg_batch_size, embed_dim)

        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)                       # (kg_batch_size, relation_dim)
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)               # (kg_batch_size, relation_dim)
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)               # (kg_batch_size, relation_dim)

        # Equation (1)
        pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)     # (kg_batch_size)
        neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)     # (kg_batch_size)

        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(r_mul_h) + _L2_loss_mean(r_embed) + _L2_loss_mean(r_mul_pos_t) + _L2_loss_mean(r_mul_neg_t)
        loss = kg_loss + self.kg_l2loss_lambda * l2_loss
        return loss


    def update_attention_batch(self, h_list, t_list, r_idx):
        r_embed = self.relation_embed.weight[r_idx]
        W_r = self.trans_M[r_idx]

        h_embed = self.entity_user_embed.weight[h_list]
        t_embed = self.entity_user_embed.weight[t_list]

        # Equation (4)
        r_mul_h = torch.matmul(h_embed, W_r)
        r_mul_t = torch.matmul(t_embed, W_r)
        v_list = torch.sum(r_mul_t * torch.tanh(r_mul_h + r_embed), dim=1)
        return v_list


    def update_attention(self, h_list, t_list, r_list, relations):
        device = self.A_in.device

        rows = []
        cols = []
        values = []

        for r_idx in relations:
            index_list = torch.where(r_list == r_idx)
            batch_h_list = h_list[index_list]
            batch_t_list = t_list[index_list]

            batch_v_list = self.update_attention_batch(batch_h_list, batch_t_list, r_idx)
            rows.append(batch_h_list)
            cols.append(batch_t_list)
            values.append(batch_v_list)

        rows = torch.cat(rows)
        cols = torch.cat(cols)
        values = torch.cat(values)

        indices = torch.stack([rows, cols])
        shape = self.A_in.shape
        A_in = torch.sparse.FloatTensor(indices, values, torch.Size(shape))

        # Equation (5)
        A_in = torch.sparse.softmax(A_in.cpu(), dim=1)
        self.A_in.data = A_in.to(device)


    def calc_score(self, user_ids, item_ids, item_feas):
        """
        user_ids:  (n_users)
        item_ids:  (n_items)
        """
        all_embed = self.calc_cf_embeddings()           # (n_users + n_entities, concat_dim)
        user_embed = all_embed[user_ids]                # (n_users, concat_dim)
        item_embed = all_embed[item_ids]                # (n_items, concat_dim)
        
        if self.abs_dvn_fusion:
            item_dvn_embed = self.dvn(item_feas)
            item_embed = self.SimpleFusion(item_dvn_embed,item_embed)
        
        cf_score = torch.matmul(user_embed, item_embed.transpose(0, 1))    # (n_users, n_items)
        return cf_score


    def forward(self, *input, mode):
        if mode == 'train_cf':
            return self.calc_cf_loss(*input)
        if mode == 'train_kg':
            return self.calc_kg_loss(*input)
        if mode == 'update_att':
            return self.update_attention(*input)
        if mode == 'predict':
            return self.calc_score(*input)


