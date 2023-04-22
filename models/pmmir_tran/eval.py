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
import sys
import numpy as np
import pandas as pd
import json

from models.user import UserSim
from models.ranker import Ranker
from models.loss import TripletLoss

from models.pmmir_tran.model_transformer import RecTransformer
from models.pmmir_tran.eval_utils import evaluation


parser = argparse.ArgumentParser(description='MMCRS')

# user simulator
parser.add_argument('--data-type', type=str, default="",
                    help='dress, shoe')
parser.add_argument('--caption-model-dir', type=str, default="",
                    help='location of caption model')

# control
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed')
parser.add_argument('--top-k', type=int, default=3,
                    help='top k candidate for policy and nearest neighbors')
parser.add_argument('--test-turns', type=int, default=10,
                    help='dialog turns for testing')
parser.add_argument('--pretrained-model', type=str, default="sl-best.pt",
                    help='path to the pretrained model checkpoint')

# save
parser.add_argument('--model-dir', type=str, default="",
                    help='the path to the recsys models')
parser.add_argument('--result-folder', type=str, default="log-eval",
                    help='store the results in this folder')

args = parser.parse_args()


# initialise models and user simulators
user = UserSim(data_type=args.data_type,caption_model_dir=args.caption_model_dir)
ranker = Ranker()
model = RecTransformer(args.top_k)

# load pre-trained model
model.load_state_dict(torch.load(os.path.join(args.model_dir, args.pretrained_model), map_location=lambda storage, loc: storage))

if torch.cuda.is_available():
    model.cuda()

random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

log_file = open(os.path.join(args.model_dir,"log.txt"), 'a')
sys.stdout = log_file

_, _, _, ndcg_topk, ndcg_10, sr = evaluation("test", model, user, ranker, args.batch_size, args.top_k, args.test_turns, args.model_dir, args.result_folder)

print("Evaluation on test set:")
print("#NDCG@topk:", ndcg_topk)
print("#Success Rate:", sr)

log_file.close()