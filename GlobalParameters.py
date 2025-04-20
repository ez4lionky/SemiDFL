from argparse import ArgumentParser
from utils.dist_wm_table import get_topology


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise TypeError('Boolean value expected.')


def parse_args():
    parser = ArgumentParser(description='DFLSemi')
    parser.add_argument('--debug', type=str2bool, const=True, nargs='?',
                        default=False, help='When in the debug mode, it will not record logs')
    parser.add_argument('--adap_agg', type=str2bool, const=True, nargs='?', default=False,
                        help='Whether to use adaptive aggregation, i.e. NAWA')
    parser.add_argument('--acc_adap_agg', type=str2bool, const=True, nargs='?', default=False,
                        help='Whether to use test accuracy adaptive aggregation')
    parser.add_argument('--label_ratio', type=float, default=0.1, help='Ratio of all labeled data, in percentage')
    parser.add_argument('--non_iid_alpha', type=float, default=1.0, help='Alpha of non iid, 100 is close to iid')
    parser.add_argument('--sharp_temperature', type=float, default=0.5, help='Temperature of label sharpening')
    parser.add_argument('--beta_param', type=float, default=0.5, help='Param of lambda sampling Beta distribution')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--dlr', type=float, default=0.001, help='learning rate for diffusion')
    parser.add_argument('--guide_w', type=float, default=5.0, help='scale of guide w for conditional diffusion')
    parser.add_argument('--thre_pl', type=float, default=0.95, help='Threshold for pseudo labeling')
    parser.add_argument('--round_num', type=int, default=500, help='Number of global rounds')
    parser.add_argument('--epoch_iteration_num', type=int, default=50, help='Number of iterations per epoch')
    parser.add_argument('--test_round_interval', type=int, default=5, help='The round interval to test')
    parser.add_argument('--sample_start_round', type=int, default=50, help='The #Round start to sample dpm')
    parser.add_argument('--sample_round_interval', type=int, default=10,
                        help='Number of round interval for diffusion sampling extra data')
    parser.add_argument('--dist_wm_id', type=int, default=1, help='ID of dist weight matrix, i.e. topology')
    parser.add_argument('--fully_supervised', type=str2bool, const=True, nargs='?', default=False,
                        help='Whether is fully supervised learning, i.e. upper bound')

    parser.add_argument('--task', type=str, default='mnist', help='mnist, fashion or cifar10',
                        choices=("mnist", "fashion", "cifar10", ))
    parser.add_argument('--same_weights', type=str2bool, const=True, nargs='?', default=True,
                        help='Whether to use same weights for LMU clients when not using adaptive weighting')
    parser.add_argument('--model', type=str, default='cnn', help='cnn, resnet18',
                        choices=("cnn", "resnet18"))
    parser.add_argument('--log_dir', type=str, default='logs', help='log dir name')
    parser.add_argument('--ssl_method', type=str, default="Supervise",
                        choices=("Supervise", "Pseudo", "MixMatch", "FlexMatch", "CBAFed", "FedSSL", "DFLSemi"))
    parser.add_argument('--gen_model', type=str, default="dpm", choices=("dpm", "gan"))
    parser.add_argument('--pl_ablation', type=str, default='none', help='vanilla, apl, none',
                        choices=("vanilla", "apl", "none"))
    parser.add_argument('--fixed_pl', type=str2bool, const=True, nargs='?', default=False,
                        help='Whether generate pseudo labels at the beginning each round and keep unchanged, '
                             'False for vanilla/adaptive pseudo labeling')
    parser.add_argument('--neighbor_pl', type=str2bool, const=True, nargs='?',
                        default=False, help='Whether generate pseudo labels using the neighborhood information')
    parser.add_argument('--with_ni', type=str2bool, const=True, nargs='?',
                        default=True, help='Whether using the neighborhood information')
    parser.add_argument('--gpu_list', type=int, default=[0], nargs='+', help='The list of used gpu')
    args = parser.parse_args()
    return args


# =================== general global parameters (EDITABLE) ====================
# maximum parallel workers
num_parallel = 6
loader_worker = 6

args = parse_args()
gpu_list = args.gpu_list
log_dir = args.log_dir
debug = args.debug
adap_agg = args.adap_agg
acc_adap_agg = args.acc_adap_agg
same_weights = args.same_weights
non_iid_alpha = args.non_iid_alpha
task = args.task
model = args.model
dist_wm_id = args.dist_wm_id
ssl_method = args.ssl_method
gen_model = args.gen_model
fixed_pl = args.fixed_pl
neighbor_pl = args.neighbor_pl
with_ni = args.with_ni
label_ratio = args.label_ratio / 100
sharp_temperature = args.sharp_temperature
beta_param = args.beta_param
lr = args.lr
dlr = args.dlr
guide_w = args.guide_w
thre_pl = args.thre_pl
fully_supervised = args.fully_supervised

# number of clients, labeled, mixed, unlabeled client ids, DL weight matrix are generated by dist_wm_id
num_worker, l_client_ids, m_client_ids, u_client_ids, dist_wm = get_topology(dist_wm_id, fully_supervised)

# number of global iteration and number of local epoch per global iteration
local_epoch_num = 1
round_num = args.round_num
epoch_iteration_num = args.epoch_iteration_num
test_round_interval = args.test_round_interval
sample_start_round = args.sample_start_round
sample_round_interval = args.sample_round_interval

local_batch_size = 10
classes_n = 10
g_size_once = 50
pad_size = 4

if ssl_method == 'DFLSemi':
    adap_agg = True
    acc_adap_agg = False
    same_weights = True
    fixed_pl = True
    neighbor_pl = True
    pl_ablation = args.pl_ablation

train_dpm_loop_num = 200 if gen_model == 'dpm' else 400
