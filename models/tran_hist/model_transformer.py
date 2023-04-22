import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import math

from models.utils import Norm, Encoder


class RecTransformer(nn.Module):
    def __init__(self, top_k):
        super().__init__()
        
        self.dim_size = 512
        self.layer_num =  6
        
        # image & text input
        self.text_norm = Norm(self.dim_size)
        self.text_linear = nn.Linear(in_features=self.dim_size, out_features=self.dim_size, bias=True)
        
        self.image_norm = Norm(self.dim_size)
        self.img_linear = nn.Linear(in_features=self.dim_size, out_features=self.dim_size, bias=True)
        
        # history encoder
        self.tran_past = Encoder(
            d_model=self.dim_size, N_layers=self.layer_num)
        
        self.out_linear_past = nn.Linear(self.dim_size, self.dim_size, bias=True)
        
        self.hist_vectors_past = []
        
        # state tracker
        self.tran = Encoder(
            d_model=self.dim_size, N_layers=self.layer_num)
        
        self.out_linear = nn.Linear(self.dim_size, self.dim_size, bias=True)
        
        self.hist_vectors = []
        
        self.sp_norm = Norm(self.dim_size)
        self.sp_token = nn.Parameter(
            torch.zeros(size=(1, self.dim_size)), requires_grad=False)

        self.init_parameters()

    def init_parameters(self):
        return

    def init_hist(self):
        self.hist_vectors.clear()
        self.hist_vectors_past.clear()
        return
    
#     # fine-tuning the policy with rl
#     def set_rl_mode(self):
#         self.train()
#         for param in self.img_linear.parameters():
#             param.requires_grad = False
#         return

#     def clear_rl_mode(self):
#         for param in self.img_linear.parameters():
#             param.requires_grad = True
#         return

    def forward_text(self, text_input):
        text_emb = F.normalize(text_input, dim=-1)
        return text_emb
    
    def forward_image(self, image_input):
        image_emb = F.normalize(image_input, dim=-1)
        return image_emb

#     def forward_text(self, text_input):
#         text_emb = self.text_linear(text_input)
#         text_emb = self.text_norm(text_emb)
#         return text_emb
    
#     def forward_image(self, image_input):
#         image_emb = self.img_linear(image_input)
#         image_emb = self.image_norm(image_emb)
#         return image_emb

    def get_sp_emb(self, batch_size):
        with torch.no_grad():
            sp_emb = self.sp_token.expand(
                size=(batch_size, 1, self.dim_size))
            
        sp_emb = self.sp_norm(sp_emb)
        return sp_emb
    
    def init_forward(self, history_input):
        batch_size = history_input.size(0)
        image_emb = self.forward_image(history_input)
        
        # # special token
        # sp_emb = self.get_sp_emb(batch_size)
        # self.hist_vectors.append(sp_emb)
        
        # image part
        # B x 1 x H
        # image_emb = image_emb.unsqueeze(dim=1)
        # print(image_emb.size())
        self.hist_vectors.append(image_emb)
        self.hist_vectors_past.append(image_emb)
        
        # state tracker
        full_input = torch.cat(self.hist_vectors_past, dim=1)
        outs = self.tran_past(full_input)
        outs = self.out_linear_past(torch.tanh(outs.mean(dim=1)))
        
        return outs

    # input:
    #   text: B x H
    #   image: B x H
    #   hist: B x L x H
    
    def merge_forward(self, image_input, text_input):
        batch_size = text_input.size(0)
        text_emb = self.forward_text(text_input)
        image_emb = self.forward_image(image_input)
        
        # special token
        sp_emb = self.get_sp_emb(batch_size)
        self.hist_vectors.append(sp_emb)
        
        # text part
        # B x 1 x H
        text_emb = text_emb.unsqueeze(dim=1)
        # # print(text_emb.size())
        self.hist_vectors.append(text_emb)

        # image part
        # B x 1 x H
        image_emb = image_emb.unsqueeze(dim=1)
        # print(image_emb.size())
        self.hist_vectors.append(image_emb)
        
        # state tracker
        full_input = torch.cat(self.hist_vectors, dim=1)
        outs = self.tran(full_input)
        outs = self.out_linear(torch.tanh(outs.mean(dim=1)))

        return outs
    
    def update_rep(self, all_input, batch_size=128):
        feat = torch.Tensor(all_input.size(0), self.dim_size)

        if torch.cuda.is_available():
            feat = feat.cuda()

        for i in range(1, math.ceil(all_input.size(0) / batch_size)):
            x = all_input[(i-1)*batch_size:(i*batch_size)]
            if torch.cuda.is_available():
                x = x.cuda()
            with torch.no_grad():
                out = self.forward_image(x)
            feat[(i-1)*batch_size:i*batch_size].copy_(out.data)

        if all_input.size(0) % batch_size > 0:
            x = all_input[-(all_input.size(0) % batch_size)::]
            if torch.cuda.is_available():
                x = x.cuda()
            with torch.no_grad():
                out = self.forward_image(x)
            feat[-(all_input.size(0) % batch_size)::].copy_(out.data)
        
        return feat