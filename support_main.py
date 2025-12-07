import os
import argparse
import datetime
import random
import numpy as np
import torch
from src.Load import load_data_support
from src.SUPPORT import Model_Net
from src.Train import run_model
import warnings
warnings.filterwarnings("ignore")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='SUPPORT', help='name of model')
    parser.add_argument('--dataset', type=str, default='TF', help='name of dataset')
    parser.add_argument('--Eval_layers', nargs='+', default=['Aarhus_1', 'small_world'], help='which network layer(s), e.g. Aarhus_1,scale_free,...')

    parser.add_argument('--learning_rate', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--ifDecay', action='store_false', help='If true, the learning rate decays (default true). If false, the learning rate is constant.')

    parser.add_argument('--epochs', type=int, default=100, help='max epochs')
    parser.add_argument('--EarlyStop', action='store_true', help='Default is false. If true, enable early stopping strategy')
    parser.add_argument('--patience', type=int, default=20, help='early stop')
    parser.add_argument('--best_metric', type=str, default='aupr', help='Primary evaluation metric used for early stopping (e.g., auc, ap, pr_auc).')

    parser.add_argument('--gcn_type', type=str, default='JK_GCN', help='e.g., GCN, JK_GCN')
    parser.add_argument('--gcn_layer', type=int, default=3, help='e.g., 2, 3, 4')

    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train data')
    parser.add_argument('--dim', type=int, default=16, help='dims of node embedding')

    parser.add_argument('--epsilon', type=int, default=7, help='epsilon of PGD')
    parser.add_argument('--alpha', type=int, default=3, help='alpha of PGD')
    parser.add_argument('--disturb_t', type=int, default=4, help='number of disturbances of PGD')

    parser.add_argument('--set_seed', type=int, default=42, help='random seed')
    parser.add_argument('--onlyTest', action='store_true', help='Default is false. If true, try to load an existing model.')
    parser.add_argument('--log', type=str, default='./log/', help='record file path')

    args = parser.parse_args()
    return args

def run_support():

    args = get_args()
    setup_seed(seed=args.set_seed)
    log_path = './log/'
    os.makedirs(log_path, exist_ok=True)
    save_path = './save/'
    os.makedirs(save_path, exist_ok=True)

    print('***************************')
    print('The program starts running.')
    print('***************************')
    args.log = log_path + args.dataset + '-Result-R1.txt'
    print(args)

    begin = datetime.datetime.now()
    print('Start time ', begin)
    time = str(begin.year) + '-' + str(begin.month) + '-' + str(begin.day) + '-' + str(begin.hour) + '-' + str(
        begin.minute) + '-' + str(begin.second)
    log = open(args.log, 'a', encoding='utf-8')
    write_infor = '\nStart time: ' + time + '\n'
    log.write(write_infor)
    write_infor = ', '.join([f"{k}: {v}" for k, v in vars(args).items()]) + '\n'
    log.write(write_infor)
    log.close()

    # load data
    train_loader, valid_loader, test_loader, gcn_data, network_numbers = load_data_support(args.dataset, args.batch_size, args.set_seed)
    # load model
    model = Model_Net(embedding_dim=args.dim, layer_number=network_numbers, gcn_data=gcn_data, gcn_type=args.gcn_type, gcn_layer=args.gcn_layer)

    if torch.cuda.is_available():
        model = model.cuda()
        for i in range(network_numbers):
            gcn_data[i].x = gcn_data[i].x.cuda()
            gcn_data[i].edge_index = gcn_data[i].edge_index.cuda()

    run_model(train_loader, valid_loader, test_loader, model, args)
    end = datetime.datetime.now()
    print('End time ', end)
    print('Run time ', end - begin)

if __name__ == '__main__':
    run_support()
