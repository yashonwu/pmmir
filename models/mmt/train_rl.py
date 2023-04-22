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
import torch.nn.functional as F
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

from models.mmt.model_transformer import RecTransformer
from models.mmt.eval_utils import evaluation
from models.mmt.train_utils import train_sl

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
parser.add_argument('--gamma', type=float, default=0.2, metavar='gamma',
                    help='discount factor')
parser.add_argument('--neg_num', type=int, default=5, metavar='neg_num',
                    help='number of negative samples')

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

# pre-trained
parser.add_argument('--pretrained-model', type=str, default="sl-best.pt",
                    help='path to pretrained sl model')
args = parser.parse_args()


# get log prob
def dist(state, user_img_idx, top_k_act_img_idx, memory_act_img_idx):
    if torch.cuda.is_available():
        user_img_idx = user_img_idx.cuda()
        top_k_act_img_idx = top_k_act_img_idx.cuda()
    target_emb = ranker.feat[user_img_idx]
    act_emb = ranker.feat[top_k_act_img_idx]

    # compute the log prob for candidates
    dist_action = []
    # dist = -torch.sum((state - target_emb) ** 2, dim=1)
    dist = torch.sum(state * target_emb, dim=1)
    dist_action.append(dist)

    for i in range(args.neg_num):
        neg_img_idx = torch.LongTensor(args.batch_size)
        # user.sample_idx(neg_img_idx)
        
        for j in range(args.batch_size):
            rand_idx = random.randint(0, memory_act_img_idx.size(1) - 1)
            neg_img_idx[j] = memory_act_img_idx[j,rand_idx]

        neg_emb = ranker.feat[neg_img_idx]
        
        # dist = -torch.sum((state - neg_emb) ** 2, dim=1)
        dist = torch.sum(state * neg_emb, dim=1)
        dist_action.append(dist)

    dist_action = torch.stack(dist_action, dim=1)
    label_idx = torch.LongTensor(args.batch_size).fill_(0)
    if torch.cuda.is_available():
        label_idx = label_idx.cuda()
    
    log_prob = torch.nn.functional.cross_entropy(input=dist_action, target=label_idx,reduction='none')
    sum_reward = torch.zeros(args.batch_size)
    if torch.cuda.is_available():
        sum_reward = sum_reward.cuda()
    for i in range(args.top_k):
        # reward = torch.sum((act_emb[:,i] - target_emb) ** 2, dim=1) #.mean()
        reward = torch.sum(act_emb[:,i]*target_emb, dim=1) #.mean()
        sum_reward = sum_reward+reward
    return log_prob, sum_reward

def train_rl(epoch, optimizer):
    print('train epoch #{}'.format(epoch))
    # model.set_rl_mode()
    model.train()
    triplet_loss.train()
    split= 'train'
    all_input = user.feature_input
    all_seq = user.train_seq
    dialog_turns = args.train_turns
    batch_size = args.batch_size

    user_img_idx = torch.LongTensor(args.batch_size)
    top_k_act_img_idx = torch.LongTensor(args.batch_size,args.top_k)
    implicit_act_img_idx = torch.LongTensor(args.batch_size,args.top_k-1)
    history_idx = torch.LongTensor(args.batch_size,len(all_seq[0]["item_log"]))
    num_epoch = math.ceil(len(all_seq) / args.batch_size)
    total_batch = num_epoch

    for batch_idx in range(1, num_epoch + 1):
        start = time.time()

        # sample target images and first turn feedback images
        # user.sample_idx(user_img_idx, split)
        # user.sample_target_idx(user_img_idx, split, batch_idx, args.batch_size, num_epoch)
        _,_ = user.sample_target_idx_history(user_img_idx, split, batch_idx, batch_size, total_batch, history_idx)
        user.sample_k_idx(top_k_act_img_idx, top_k=args.top_k)

        feat = model.update_rep(all_input)
        ranker.update_rep(feat)
        
        # clean up dialog history tracker
        model.init_hist()

        outs = []
    
        loss_sum = 0
        prob_list= []
        reward_list = []
        prob_init_list = []
        hist_feedback = []
        
        # memory of act_img_idx
        memory_act_img_idx = torch.LongTensor(args.batch_size,1)
        memory_act_img_idx=torch.reshape(top_k_act_img_idx, (args.batch_size, args.top_k))
        if torch.cuda.is_available():
            memory_act_img_idx = memory_act_img_idx.cuda()
            
        # memory of implicit_act_img_idx
        implicit_memory_act_img_idx = torch.LongTensor(args.batch_size,1)

        for k in range(dialog_turns):
            # find the most similar item for positive feedback
            p_act_img_idx, p_position, n_act_img_idx, n_position, implicit_act_img_idx = ranker.nearest_neighbor_selector_implicit(user_img_idx, top_k_act_img_idx)
            act_input = all_input[p_act_img_idx]
            
            # negative feedback history
            if k==0:
                implicit_memory_act_img_idx=torch.reshape(implicit_act_img_idx, (args.batch_size, args.top_k-1))
                if torch.cuda.is_available():
                    implicit_memory_act_img_idx = implicit_memory_act_img_idx.cuda()
            # else:
            #     implicit_memory_act_img_idx = torch.cat((implicit_memory_act_img_idx, torch.reshape(implicit_act_img_idx, (args.batch_size, args.top_k-1)).cuda()), 1)
            
            # get relative captions from user model given user target images and feedback images
            txt_input,_ = user.get_feedback(p_act_img_idx, user_img_idx)
            hist_feedback.append(txt_input)
            
            if torch.cuda.is_available():
                act_input = act_input.cuda()
                txt_input = txt_input.cuda()
            
            # update the query action vector given feedback image and text feedback in this turn
            state = model.merge_forward(act_input, txt_input)
            
            # sampling the recommendations
            top_km_act_img_idx = ranker.k_nearest_neighbors(state.data,K=args.top_k*args.train_turns)
            for i in range(args.batch_size):
                # print(i)
                k_item=0
                for j in range(args.top_k*args.val_turns):
                    # print(j)
                    if top_km_act_img_idx[i,j].cpu().numpy() in memory_act_img_idx[i,:].cpu().numpy():
                        pass
                    else:
                        top_k_act_img_idx[i,k_item]=top_km_act_img_idx[i,j]
                        if k_item==args.top_k-1:
                            break #skip the rest in top-k
                        k_item=k_item+1
            
            memory_act_img_idx = torch.cat((memory_act_img_idx, torch.reshape(top_k_act_img_idx, (args.batch_size, args.top_k)).cuda()), 1)
            
            # find the most similar item for positive feedback
            p_act_img_idx, p_position, n_act_img_idx, n_position, implicit_act_img_idx = ranker.nearest_neighbor_selector_implicit(user_img_idx, top_k_act_img_idx)
            implicit_memory_act_img_idx = torch.cat((implicit_memory_act_img_idx, torch.reshape(implicit_act_img_idx, (args.batch_size, args.top_k-1)).cuda()), 1)
            
            # log_prob, reward
            log_prob,reward = dist(state, user_img_idx, top_k_act_img_idx, implicit_memory_act_img_idx)
            
            prob_list.append(log_prob)
            reward_list.append(reward)

            ## option 1: random new actions
            # user.sample_k_idx(top_k_act_img_idx, top_k=args.top_k)
            
            # option 2: next action
            top_k_act_img_idx = top_k_act_img_idx
        
        # finish dialog and update model parameters
        optimizer.zero_grad()
        
        reversed_prob_list = reversed(prob_list)
        cum_reward = 0.
        for i, prob_temp in enumerate(reversed_prob_list):
            cum_reward = reward_list[dialog_turns-1-i] + args.gamma * cum_reward
            policy_loss = prob_temp * cum_reward
            loss_sum = policy_loss + loss_sum
        
        # for i, prob_temp in enumerate(prob_list):
        #     loss_sum = prob_temp + loss_sum

        loss_sum.mean().backward()
        optimizer.step()

        end = time.time()
        print('batch_idx:', batch_idx, '/', num_epoch, ', time elapsed:{:.2f}'.format(end - start))
    return

# initialise models and user simulators
user = UserSim(data_type=args.data_type,caption_model_dir=args.caption_model_dir)
ranker = Ranker()
model = RecTransformer(args.top_k)
triplet_loss = TripletLoss(margin=args.triplet_margin)

# load pre-trained model
model.load_state_dict(torch.load(args.pretrained_model, map_location=lambda storage, loc: storage))

if torch.cuda.is_available():
    model.cuda()

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

    train_rl(epoch, optimizer)
    
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
        torch.save(model.state_dict(), (os.path.join(args.model_dir,'rl-best.pt')).format(epoch))
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