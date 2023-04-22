from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from models.utils import Norm, Encoder
import math


class RecGRU(nn.Module):
    def __init__(self, top_k):
        super(RecGRU, self).__init__()
        
        self.dim_size = 512
        
        # image & text input
        self.text_norm = Norm(self.dim_size)
        self.text_linear = nn.Linear(in_features=self.dim_size, out_features=self.dim_size, bias=True)
        
        self.image_norm = Norm(self.dim_size)
        self.img_linear = nn.Linear(in_features=self.dim_size, out_features=self.dim_size, bias=True)
        
        # combiner
        self.fc_joint = nn.Linear(in_features=self.dim_size*2, out_features=self.dim_size, bias=False)
        
        self.rnn = nn.GRUCell(self.dim_size, self.dim_size, bias=False)
        self.head = nn.Linear(in_features=self.dim_size, out_features=self.dim_size, bias=False)

#     # fine-tuning the history tracker and policy part
#     def set_rl_mode(self):
#         self.train()
#         for param in self.img_linear.parameters():
#             param.requires_grad = False
#         return

#     def clear_rl_mode(self):
#         for param in self.img_linear.parameters():
#             param.requires_grad = True
#         return
    
#     def forward_text(self, text_input):
#         text_emb = self.text_linear(text_input)
#         text_emb = self.text_norm(text_emb)
#         return text_emb
    
#     def forward_image(self, image_input):
#         image_emb = self.img_linear(image_input)
#         image_emb = self.image_norm(image_emb)
#         return image_emb
    
    def forward_text(self, text_input):
        text_emb = F.normalize(text_input, dim=-1)
        return text_emb
    
    def forward_image(self, image_input):
        image_emb = F.normalize(image_input, dim=-1)
        return image_emb

    def init_forward(self, history_input):
        batch_size = history_input.size(0)
        image_emb = self.forward_image(history_input)
        
        for i in range(history_input.size(1)):
            x = image_emb[:,i,:]
            self.hx = self.rnn(x, self.hx)
            
        outs = self.head(self.hx)
        return outs

    def merge_forward(self, img_input, txt_input):
        x1 = self.forward_image(img_input)
        x2 = self.forward_text(txt_input)
        
        # x1 = torch.reshape(x1, (x1.size(0), x1.size(1)*x1.size(2)))
        x = torch.cat([x1, x2], dim=1)
        x = self.fc_joint(x)
        self.hx = self.rnn(x, self.hx)
        x = self.head(self.hx)
        return x
        # return F.normalize(x)

    def init_hid(self, batch_size):
        self.hx = torch.Tensor(batch_size, self.dim_size).zero_()
        return

    def detach_hid(self):
        self.hx = self.hx.data
        return
    
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

