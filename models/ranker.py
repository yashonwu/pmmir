from __future__ import print_function
import torch
from torch.autograd import Variable
import math
import numpy

class Ranker():
    def __init__(self):
        super(Ranker, self).__init__()
        return
    
    def update_rep(self, feat):
        self.feat = feat
        return

    def compute_rank(self, input, target_idx):
        # input <---- a batch of vectors
        # targetIdx <----- ground truth index
        # return rank of input vectors in terms of rankings in distance to the ground truth

        if torch.cuda.is_available():
            # input = input.cuda()
            target_idx = target_idx.cuda()
            # self.feat = self.feat.cuda()
        target = self.feat[target_idx]

        value = target - input
        value = value ** 2
        value = value.sum(1)
        rank = torch.LongTensor(value.size(0))
        for i in range(value.size(0)):
            val = self.feat - input[i].expand(self.feat.size(0), self.feat.size(1))
            val = val ** 2
            val = val.sum(1)
            rank[i] = val.lt(value[i]).sum()

        return rank

    def compute_rank_correct(self, input, target_idx, memory_idx):
        # input <---- a batch of vectors
        # targetIdx <----- ground truth index
        # return rank of input vectors in terms of rankings in distance to the ground truth

        if torch.cuda.is_available():
            # input = input.cuda()
            target_idx = target_idx.cuda()
            # self.feat = self.feat.cuda()
        target = self.feat[target_idx]
        memory = self.feat[memory_idx]

        value = target - input
        value = value ** 2
        value = value.sum(1)
        rank = torch.LongTensor(value.size(0))
        for i in range(value.size(0)):
            val = self.feat - input[i].expand(self.feat.size(0), self.feat.size(1))
            val = val ** 2
            val = val.sum(1)

            val_m = memory[i] - input[i].expand(memory[i].size(0), memory[i].size(1))
            val_m = val_m ** 2
            val_m = val_m.sum(1)

            rank[i] = val.lt(value[i]).sum()-val_m.lt(value[i]).sum()

            # print("----------------------------------")
            # print("vanilla rank:", val.lt(value[i]).sum())
            # print("memory rank:", val_m.lt(value[i]).sum())

            # there might be the same images as the target image, i.e. duplicated images in dataset
            # so the rank[i] can be -1, we need to correct this
            if rank[i]<0:
                rank[i] = 0

        return rank
    
    def compute_rank_top3(self, top_k_candidate_idx, target_idx):
        rank = torch.LongTensor(target_idx.size(0))
        for i in range(target_idx.size(0)):
            top_k_candidate = top_k_candidate_idx[i]
            target = target_idx[i]
            if target in top_k_candidate:
                idx = top_k_candidate.tolist().index(target)
            else:
                idx = 100
            rank[i] = idx
        return rank

    def nearest_neighbor(self, target):
        # L2 case
        idx = torch.LongTensor(target.size(0))
        if torch.cuda.is_available():
            target = target.cuda()
            # self.feat = self.feat.cuda()
        for i in range(target.size(0)):
            val = self.feat - target[i].expand(self.feat.size(0), self.feat.size(1))
            val = val ** 2
            val = val.sum(1)
            v, id = val.min(0)
            idx[i] = id #[0]
        return idx

    def k_nearest_neighbors(self, target, K = 5):
        idx = torch.LongTensor(target.size(0), K)
        if torch.cuda.is_available():
            target = target.cuda()
            self.feat = self.feat.cuda()

        for i in range(target.size(0)):
            val = self.feat - target[i].expand(self.feat.size(0), self.feat.size(1))
            val = val ** 2
            val = val.sum(1)
            v, id = torch.topk(val, k=K, dim=0, largest=False)
            # idx[i] = id
            idx[i].copy_(id.view(-1))
        return idx

    def nearest_neighbor_selector(self, user_img_idx, top_k_act_img_idx):
        # L2 case
        # print("feat shape:", self.feat.shape)
        target = self.feat[user_img_idx]
        # print("target shape:", target.shape)
        min_idx = torch.LongTensor(target.size(0))
        min_pos = torch.LongTensor(target.size(0))
        
        max_idx = torch.LongTensor(target.size(0))
        max_pos = torch.LongTensor(target.size(0))
        if torch.cuda.is_available():
            target = target.cuda()
            # self.feat = self.feat.cuda()
        feat = self.feat[top_k_act_img_idx]
        # print("feat shape:", feat.shape)
        for i in range(target.size(0)):
            val = feat[i] - target[i].expand(feat[i].size(0), feat[i].size(1))
            val = val ** 2
            val = val.sum(1)
            
            # most similar
            v, id = val.min(0)
            min_idx[i] = top_k_act_img_idx[i,id]
            min_pos[i] = id
            
            # leaset similar
            v, id = val.max(0)
            max_idx[i] = top_k_act_img_idx[i,id]
            max_pos[i] = id
            # print("idx[i]", idx[i])
        return min_idx, min_pos, max_idx, max_pos
    
    def nearest_neighbor_selector_implicit(self, user_img_idx, top_k_act_img_idx):
        # L2 case
        # print("feat shape:", self.feat.shape)
        target = self.feat[user_img_idx]
        # print("target shape:", target.shape)
        min_idx = torch.LongTensor(target.size(0))
        min_pos = torch.LongTensor(target.size(0))
        
        max_idx = torch.LongTensor(target.size(0))
        max_pos = torch.LongTensor(target.size(0))
        
        if torch.cuda.is_available():
            target = target.cuda()
            # self.feat = self.feat.cuda()
        feat = self.feat[top_k_act_img_idx]
        # print("feat shape:", feat.shape)
        
        implicit_idx = torch.LongTensor(target.size(0), feat.shape[1]-1)
        
        for i in range(target.size(0)):
            val = feat[i] - target[i].expand(feat[i].size(0), feat[i].size(1))
            val = val ** 2
            val = val.sum(1)
            
            # most similar
            v, id = val.min(0)
            min_idx[i] = top_k_act_img_idx[i,id]
            min_pos[i] = id
            
            # negatives
            topk_ids = list(range(feat.shape[1]))
            topk_ids.remove(id)
            implicit_idx[i] = top_k_act_img_idx[i][topk_ids]
            # implicit_idx[i] = top_k_act_img_idx[i][top_k_act_img_idx[i]!=top_k_act_img_idx[i,id]]
            
            # leaset similar
            v, id = val.max(0)
            max_idx[i] = top_k_act_img_idx[i,id]
            max_pos[i] = id
            # print("idx[i]", idx[i])
            
        return min_idx, min_pos, max_idx, max_pos, implicit_idx

