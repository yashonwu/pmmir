# import os, sys
# from sys import path
# sys.path.insert(0, os.getcwd())
import torch
import random
import numpy as np
import pickle
from torch.autograd import Variable
import json
# from transformers import AutoTokenizer, AutoModel
import clip

from captioning import captioner
from captioning.utils.resnet_utils import myResnet
import captioning.utils.resnet as resnet

from torch.nn.utils.rnn import pad_sequence

# two functions are tested in this script:
# fc, att = captioner.compute_img_feat_batch(images)
# fc: N x 2048, att: N x 7 x 7 x2048
# and
# seq_id, sentences = captioner.gen_caption_from_feat(feat_tuple_target, feat_tuple_target)
# seq_id: N x 8, sentences: N string sentences

class UserSim:
    def __init__(self, data_type='', caption_model_dir=''):
        self.data_type = data_type
        
        # load trained captioner model
        params = {}
        params['model'] = 'resnet101'
        params['model_root'] = 'imagenet_weights'
        params['att_size'] = 7
        params['beam_size'] = 1
        self.captioner_relative = captioner.Captioner(is_relative= True, model_path= caption_model_dir, image_feat_params= params)
        self.captioner_relative.opt['use_att'] = True
        self.vocabSize = self.captioner_relative.get_vocab_size()
        
        random.seed(42)
        
        self.clip_model, preprocess = clip.load("ViT-B/32")
        self.clip_model.cuda().eval()
        
        from .data_utils import get_embeddings, get_data_splits, get_docnos
        self.fc_input, self.att_input, self.feature_input = get_embeddings(data_type)
        
        self.u_i_seq, self.train_seq, self.val_seq, self.test_seq = get_data_splits(data_type)
        self.docnos = get_docnos(data_type)
        
        self.all_imgs_size = len(self.feature_input)
        
        self.train_size = len(self.train_seq)
        self.val_size = len(self.test_seq)
        self.test_size = len(self.test_seq)

        print('init. done!\n#img: {}/ {} / {}'.format(self.train_size, self.val_size, self.test_size))
        print('number of images in total:', self.all_imgs_size)
        print('use cuda:', torch.cuda.is_available())
        return

    # sample images randomly for random recommendations
    def sample_idx(self, img_idx):

        for i in range(img_idx.size(0)):
            img_idx[i] = random.randint(0, self.all_imgs_size - 1)
        return
    
    def sample_k_idx(self, img_idx, top_k):
        
        for i in range(img_idx.size(0)):
            img_idx[i] = torch.Tensor(random.sample(range(0, self.all_imgs_size - 1), img_idx.size(1)))
            
        if torch.cuda.is_available():
            img_idx = img_idx.cuda()
        return
    
    def name_2_idx(self, name):
        idx = self.docnos.index(name)
        return idx
    
    # sample users
    def sample_target_idx_history(self, img_idx, split, batch_idx, batch_size, num_epoch, history_idx):
        if split == 'train':
            split_size = self.train_size
            user_seq = self.train_seq
        elif split == 'val':
            split_size = self.val_size
            user_seq = self.val_seq
        elif split == 'test':
            split_size = self.test_size
            user_seq = self.test_seq

        temp_history_idx = []
        user_name_list = []
        if batch_idx==num_epoch:
            left=np.arange(split_size)[(batch_idx-1)*batch_size:]
            for i in range(img_idx.size(0)):
                if i<len(left):
                    target_name = user_seq[left[i]]["item_target"]
                    target_idx = self.name_2_idx(target_name)
                    img_idx[i] = torch.tensor(target_idx)
                    
                    log_name = user_seq[left[i]]["item_log"]
                    log_idx = [self.name_2_idx(x) for x in log_name]
                    # history_idx[i] = torch.tensor(log_idx)
                    temp_history_idx.append(log_idx)
                    
                    # user ids
                    user_name = user_seq[left[i]]["customer_id"]
                    user_name_list.append(user_name)
                else:
                    j = random.randint(0, split_size - 1)
                    
                    target_name = user_seq[j]["item_target"]
                    target_idx = self.name_2_idx(target_name)
                    img_idx[i] = torch.tensor(target_idx)
                    
                    log_name = user_seq[j]["item_log"]
                    log_idx = [self.name_2_idx(x) for x in log_name]
                    # history_idx[i] = torch.tensor(log_idx)
                    temp_history_idx.append(log_idx)
                    
                    # user ids
                    user_name = user_seq[j]["customer_id"]
                    user_name_list.append(user_name)

        else:
            range_idx = np.arange(split_size)[(batch_idx-1)*batch_size:batch_idx*batch_size]
            for i in range(img_idx.size(0)):
                target_name = user_seq[range_idx[i]]["item_target"]
                target_idx = self.name_2_idx(target_name)
                img_idx[i] = torch.tensor(target_idx)
                
                log_name = user_seq[range_idx[i]]["item_log"]
                log_idx = [self.name_2_idx(x) for x in log_name]
                # history_idx[i] = torch.tensor(log_idx)
                temp_history_idx.append(log_idx)
                
                # user ids
                user_name = user_seq[range_idx[i]]["customer_id"]
                user_name_list.append(user_name)
                
        temp_input_list = []
        for i in range(img_idx.size(0)):
            temp_input = self.feature_input[temp_history_idx[i]]
            temp_input_list.append(temp_input)
        history_input = pad_sequence(temp_input_list, batch_first=True)
        
        # print(history_input.size())
        return history_input, user_name_list

    def get_image_name(self, img_idx):
        imgs_name = [self.docnos[x] for x in img_idx]
        return imgs_name

    def get_top_k_image_name(self, img_idx):
        imgs_name = [[self.docnos[y] for y in x] for x in img_idx]
        return imgs_name

    def get_feedback(self, act_idx, user_idx):
        fc = self.fc_input
        att = self.att_input

        batch_size = user_idx.size(0)
        
        # load embeddings for the batch
        act_fc = fc[act_idx]
        act_att = att[act_idx]
        user_fc = fc[user_idx]
        user_att = att[user_idx]
        
        act_fc = torch.Tensor(act_fc)
        act_att = torch.Tensor(act_att)
        user_fc = torch.Tensor(user_fc)
        user_att = torch.Tensor(user_att)
        
        if torch.cuda.is_available():
            act_fc = act_fc.cuda()
            act_att = act_att.cuda()
            user_fc = user_fc.cuda()
            user_att = user_att.cuda()
            
        # positive feeback
        with torch.no_grad():
            seq_label, sents_label = self.captioner_relative.gen_caption_from_feat((user_fc,user_att),
                                                                               (act_fc,act_att))

        text_tokens = clip.tokenize(sents_label, truncate=True).cuda()
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens).float()

        return text_features, sents_label

    
    
    
