from __future__ import print_function
import os, sys
from sys import path
sys.path.insert(0, os.getcwd())
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import math
import random
import time
import numpy as np
import pandas as pd
import json
import glob
import shutil

from models.user import UserSim
from models.ranker import Ranker
from models.loss import TripletLoss

from models.pmmir.model_gru import RecGRU
from models.pmmir.eval_utils import evaluation
from models.pmmir.train_utils import train_sl

parser = argparse.ArgumentParser(description='MMCRS')

# user simulator
parser.add_argument('--data-type', type=str, default="",
                    help='dress, shoe')
parser.add_argument('--caption-model-dir', type=str, default="",
                    help='location of caption model')

# learning
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training/evaluation')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--triplet-margin', type=float, default=0.1, metavar='EV',
                    help='triplet loss margin ')

# control
parser.add_argument('--top-k', type=int, default=1,
                    help='top k candidate for policy and nearest neighbors')
parser.add_argument('--train-turns', type=int, default=10,
                    help='dialog turns for training')
parser.add_argument('--val-turns', type=int, default=10,
                    help='dialog turns for validation')

# save
parser.add_argument('--model-dir', type=str, default="",
                    help='the path to the recsys models')
parser.add_argument('--result-folder', type=str, default="log-val",
                    help='store the results in this folder')
args = parser.parse_args()


# initialise models and user simulators
user = UserSim(data_type=args.data_type,caption_model_dir=args.caption_model_dir)
ranker = Ranker()
model = RecGRU(args.top_k)
triplet_loss = TripletLoss(margin=args.triplet_margin)

if torch.cuda.is_available():
    model.cuda()
    triplet_loss.cuda()

# random seed setting
random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

# optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
                          weight_decay=1e-8)

# recsys model dir
if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)

# validation dir
temp_val_log_folder = os.path.join(args.model_dir, args.result_folder)
if not os.path.exists(temp_val_log_folder):
    os.makedirs(temp_val_log_folder)
best_val_log_folder = os.path.join(args.model_dir,"log-val-best")
if not os.path.exists(best_val_log_folder):
    os.makedirs(best_val_log_folder)

best_perform = 0.0
best_epoch = 0

log_file = open(os.path.join(args.model_dir,"log.txt"), 'a')
# sys.stdout = log_file

early_stop_max = 5
early_stop_count = 0

for epoch in range(1, args.epochs + 1):
    # log_file.write('train epoch #{}\n'.format(epoch))
    print('train epoch #{}'.format(epoch), file=log_file, flush=True)

    # ------- train
    start = time.time()

    train_sl(epoch, optimizer, triplet_loss, model, user, ranker, args.batch_size, args.top_k, args.train_turns)
    
    end = time.time()
    print('Time elapsed for training:{:.2f}'.format(end - start), file=log_file, flush=True)

    # ------- validate
    start = time.time()

    _, ndcg_mean, sr_mean, _, ndcg, sr = evaluation('val', model, user, ranker, args.batch_size, args.top_k, args.val_turns, args.model_dir, args.result_folder)
    print('Mean NDCG@10:{:.4f}; Mean SR:{:.4f}'.format(ndcg_mean,sr_mean), file=log_file, flush=True)
    # log_file.write('NDCG@10 on validation set:{}\n'.format(ndcg))
    # log_file.write('SR on validationset:{}\n'.format(sr))
    # log_file.write('Mean NDCG@10:{:.4f}; Mean SR:{:.4f}\n'.format(ndcg_mean,sr_mean))

    temp_val_log_files = glob.glob(temp_val_log_folder+"/*")
    best_val_log_files = glob.glob(best_val_log_folder+"/*")

    epoch_perform = ndcg_mean

    if epoch_perform > best_perform:
        best_perform = epoch_perform
        best_epoch = epoch
        torch.save(model.state_dict(), (os.path.join(args.model_dir,'sl-best.pt')).format(epoch))
        print("Best performance on validation set:{:.4f}; best epoch:{}".format(best_perform, best_epoch), file=log_file, flush=True)
        # log_file.write("Best performance on validation set:{:.4f}; best epoch:{}\n".format(best_perform, best_epoch))

        # empty the best_val_log folder
        for f in best_val_log_files:
            os.remove(f)
        # copy the best temp_val_log_files to the best folder
        for f in temp_val_log_files:
            shutil.copy(f, best_val_log_folder)

    # empty the temp_val_log folder
    for f in temp_val_log_files:
        os.remove(f)
    
    end = time.time()
    print('Time elapsed for validation:{:.2f}'.format(end - start), file=log_file, flush=True)
    
    if early_stop_count >= early_stop_max:
        break
    
    # torch.save(model.state_dict(), (args.model_dir+'/sl-{}.pt').format(epoch))

log_file.close()