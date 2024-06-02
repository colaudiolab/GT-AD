import argparse
import os
import torch
# from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
# from exp.exp_imputation import Exp_Imputation
# from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_ad import Exp_Anomaly_Detection
# from exp.exp_classification import Exp_Classification
import random
import numpy as np


parser = argparse.ArgumentParser(description='MyModel')

# basic config
#NIPS_TS_GECCO
parser.add_argument('--task_name', type=str, default='Anomaly_Detection')
parser.add_argument('--model', type=str, default='MyModel')
parser.add_argument('--data', type=str,  default='NIPS_TS_GECCO', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/NIPS_TS_GECCO/', help='root path of the data file')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# anomaly detection task
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--picture', type=bool, default=False, choices=[True, False])
parser.add_argument('--win_size', type=int, default=100, help='input sequence length')
parser.add_argument('--anomaly_ratio', type=float, default=8, help='prior anomaly ratio (%)')

# gpt model define
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
parser.add_argument('--enc_in', type=int, default=9, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=9, help='decoder input size')
parser.add_argument('--c_out', type=int, default=9, help='output size')
parser.add_argument('--d_model', type=int, default=768, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=8, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')

# optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=5, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=2, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')


# patching
parser.add_argument('--patch_size', type=int, default=1)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--gpt_layers', type=int, default=6)
parser.add_argument('--ln', type=int, default=0)
parser.add_argument('--mlp', type=int, default=0)
parser.add_argument('--weight', type=float, default=0)
parser.add_argument('--percent', type=int, default=5)

args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)
Exp = Exp_Anomaly_Detection

if args.mode == 'train':
    for ii in range(args.itr):
        # setting record of experiments
        setting = 'train_{}_{}_{}_{}_{}'.format(
            args.task_name,
            args.model,
            args.data,
            args.win_size,
            ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
        torch.cuda.empty_cache()

if args.mode == 'test':
    for ii in range(args.itr):
        setting = 'test_{}_{}_{}_{}'.format(
            args.task_name,
            args.model,
            args.data,
            args.win_size)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
        torch.cuda.empty_cache()
