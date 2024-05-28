import argparse


def parse_cke_args():
    parser = argparse.ArgumentParser(description="Run CKE.")

    parser.add_argument('--seed', type=int, default=2019,
                        help='Random seed.')

    parser.add_argument('--dataset', nargs='?', default='amazon-book',
                        help='Choose a dataset from {yelp2018, last-fm, amazon-book}')
    parser.add_argument('--data_dir', nargs='?', default='data/',
                        help='Input data path.')

    parser.add_argument('--use_pretrain', type=int, default=0,
                        help='0: No pretrain, 1: Pretrain with the learned embeddings, 2: Pretrain with stored model.')
    parser.add_argument('--pretrain_embedding_dir', nargs='?', default='datasets/pretrain/',
                        help='Path of learned embeddings.')
    parser.add_argument('--pretrain_model_path', nargs='?', default='trained_model/model.pth',
                        help='Path of stored model.')

    parser.add_argument('--cf_batch_size', type=int, default=1024,
                        help='CF batch size.')
    parser.add_argument('--kg_batch_size', type=int, default=2048,
                        help='KG batch size.')
    parser.add_argument('--test_batch_size', type=int, default=10000,
                        help='Test batch size (the user number to test every batch).')

    parser.add_argument('--embed_dim', type=int, default=64,
                        help='User / item / entity Embedding size.')
    parser.add_argument('--relation_dim', type=int, default=64,
                        help='Relation Embedding size.')

    parser.add_argument('--kg_l2loss_lambda', type=float, default=1e-5,
                        help='Lambda when calculating KG l2 loss.')
    parser.add_argument('--cf_l2loss_lambda', type=float, default=1e-5,
                        help='Lambda when calculating CF l2 loss.')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--n_epoch', type=int, default=1000,
                        help='Number of epoch.')
    parser.add_argument('--stopping_steps', type=int, default=10,
                        help='Number of epoch for early stopping')

    parser.add_argument('--print_every', type=int, default=1,
                        help='Iter interval of printing CF loss.')
    parser.add_argument('--evaluate_every', type=int, default=1,
                        help='Epoch interval of evaluating CF.')

    parser.add_argument('--Ks', nargs='?', default='[1, 5, 10, 15]',
                        help='Calculate metric@K when evaluating.')

    parser.add_argument('--add_dvn',type = bool, default = False, 
                        help = 'add dvn or not.')
    parser.add_argument('--add_pseudo', type = bool, default = False)
    parser.add_argument('--dvn_dropout_rate', type=float, default =0.1, help= 'dvn dropout rate')
    parser.add_argument('--num_ps', type = int, default= 1347, help = 'number of ps samples')

    parser.add_argument('--abs_ps_a', type = bool, default = False, help = 'drop tau ego' )
    parser.add_argument('--abs_ps_b', type = bool, default = False, help = 'drop meta path' )
    parser.add_argument('--abs_dvn_up', type = bool, default = False, help = 'drop dvn up' )
    parser.add_argument('--abs_dvn_pop', type = bool, default = False, help = 'drop dvn pop' )
    parser.add_argument('--abs_dvn_fusion', type = bool, default = False, help = 'drop weighted loss')
    parser.add_argument('--score_mode', type = bool, default = False, help = 'generate score table or not')


    args = parser.parse_args()

    save_dir = 'Logs/CKE/{}/embed-dim{}_relation-dim{}_lr{}_cfl2{}_kgl2{}_pretrain{}_ps{}_{}_{}_{}_{}_{}/'.format(
        args.dataset, args.embed_dim, args.relation_dim, args.lr, args.cf_l2loss_lambda, args.kg_l2loss_lambda, args.use_pretrain, args.num_ps, args.abs_ps_a, args.abs_ps_b, args.abs_dvn_up, args.abs_dvn_pop, args.abs_dvn_fusion )
    args.save_dir = save_dir

    return args


