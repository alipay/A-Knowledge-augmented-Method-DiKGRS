 
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="VRKG")

    # ===== dataset ===== #
    parser.add_argument("--dataset", nargs="?", default="movie", help="Choose a dataset:[last-fm, movie]")
    parser.add_argument(
        "--data_path", nargs="?", default="/ossfs/workspace/data/", help="Input data path."
    )

    # ===== train ===== #
    parser.add_argument('--epoch', type=int, default=1000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--dim', type=int, default=64, help='embedding size')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 regularization weight')
    parser.add_argument('--kg_l2loss_lambda', type=float, default=1e-5,
                        help='Lambda when calculating KG l2 loss.')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument("--inverse_r", type=bool, default=True, help="consider inverse relation or not")
    parser.add_argument('--sim_regularity', type=float, default=1e-4, help='regularization weight for latent factor')
    parser.add_argument("--node_dropout", type=bool, default=True, help="consider node dropout or not")
    parser.add_argument("--node_dropout_rate", type=float, default=0.5, help="ratio of node dropout")
    parser.add_argument("--mess_dropout", type=bool, default=True, help="consider message dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of node dropout")
    parser.add_argument("--batch_test_flag", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--channel", type=int, default=64, help="hidden channels for model")
    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument('--Ks', nargs='?', default='[1, 5, 10, 15]', help='Output sizes of every layer')

    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    parser.add_argument("--n_virtual", type=int, default=3, help="number of latent factor for user favour")
    parser.add_argument("--ind", type=str, default='distance', help="Independence modeling: mi, distance, cosine")
    parser.add_argument("--n_factors", type=int, default=4, help="number of latent factor for user favour")

    # ===== relation context ===== #
    parser.add_argument('--context_hops', type=int, default=2, help='number of context hops')
    parser.add_argument('--n_iter', type=int, default=3, help='number of n_iter')

    # ===== save model ===== #
    parser.add_argument("--save", type=bool, default=False, help="save model or not")
    parser.add_argument("--out_dir", type=str, default="/ossfs/workspace/", help="output directory for model")

    parser.add_argument('--add_pseudo', type = bool, default = False, help = 'add pseudo entries or not')
    parser.add_argument('--add_dvn', type = bool, default = False, help = 'add DVN or not')
    parser.add_argument('--fusion_gate', type = bool, default= False, help = 'add FusionGate or simple NN')
    parser.add_argument('--dvn_hidden_dim',  nargs='?', default = '[]',help = 'hidden size of DVN hidden layer')
    parser.add_argument('--dvn_dropout_rate', type=float, default =0.1, help= 'dvn dropout rate')
    parser.add_argument('--pseudo_weight', type=float, default=0.75, help='pseudo rating loss weights')
    parser.add_argument('--log_fname', nargs = '?', default='VRKG4Rec_movie_TrainLog0.txt', help = 'log file fname' )
    parser.add_argument('--raw_loss', type = bool, default= False,  help = 'using raw loss or pseudo weighted loss')
    parser.add_argument('--dvn_threshold', type = int, default = 2, help = 'threshold of adding dvn net')
    parser.add_argument('--num_ps', type= int, default = 1846, help = 'number of adding pseudo samples')
    parser.add_argument('--random_seed', type= int, default = 2021, help = 'random seet setting')
    parser.add_argument('--backbone', nargs="?", default = 'VRKG', help = 'VRKG,KGIN')

    parser.add_argument('--abs_ps_a', type = bool, default = False, help = 'drop tau ego' )
    parser.add_argument('--abs_ps_b', type = bool, default = False, help = 'drop meta path' )
    parser.add_argument('--abs_dvn_up', type = bool, default = False, help = 'drop dvn up' )
    parser.add_argument('--abs_dvn_pop', type = bool, default = False, help = 'drop dvn pop' )
    parser.add_argument('--abs_dvn_fusion', type = bool, default = False, help = 'drop weighted loss')
    parser.add_argument('--score_mode', type = bool, default= False, help = 'generate scoring table with pretrained model'  )




    
    

    return parser.parse_args()
