from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from transformers import GPT2ForSequenceClassification
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import BertTokenizer, BertModel
from einops import rearrange
import math
from layers.Embed import DataEmbedding, DataEmbedding_wo_time

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class GPT(nn.Module):
    def __init__(self, args):
        super(GPT, self).__init__()
        self.win_size = args.win_size
        self.gpt_layers = args.gpt_layers
        self.input_c = args.enc_in
        # patch
        self.embed_dim = 768
        if args.data == 'NIPS_TS_Swan':
            self.patch_len = 10
            self.patch_num = 10
            self.stride_len = 10
        elif args.data == 'NIPS_TS_GECCO':
            self.patch_len = 10
            self.patch_num = 10
            self.stride_len = 10
        else:
            self.patch_len = 10
            self.patch_num = 10
            self.stride_len = 10
        self.pos_drop = nn.Dropout(0.0)
        self.pos_embed = PositionalEmbedding(d_model=self.embed_dim)
        self.time_token = nn.Conv1d(self.input_c, self.input_c * self.embed_dim,
                                    kernel_size=self.patch_len, stride=self.stride_len, groups=self.input_c)

        self.gpt2 = GPT2Model.from_pretrained('./gpt2', output_attentions=True, output_hidden_states=True)
        self.gpt2.h = self.gpt2.h[:self.gpt_layers]
        
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name: # or 'mlp' in name:
                param.requires_grad = True
            elif 'mlp' in name and args.mlp == 1:
                param.requires_grad = True
            else:
                param.requires_grad = False

        if args.use_gpu:
            device = torch.device('cuda:{}'.format(0))
            self.gpt2.to(device=device)

        self.out_layer = nn.Linear(self.patch_num*self.embed_dim,1080)

    def forward(self, x):
        B, L, C = x.shape
        x = x.transpose(1,2)
        # 求time_token
        x_time = x
        x_time_token = self.time_token(x_time).reshape(B, C, self.embed_dim,-1)
        x_time_token = x_time_token.transpose(2, 3).reshape(B*C,-1,self.embed_dim)
        x_time_token = self.pos_drop(x_time_token + self.pos_embed(x_time_token))
        
        outputs = self.gpt2(inputs_embeds=x_time_token).last_hidden_state
        outputs = self.out_layer(outputs.reshape(B*C, -1))
        return outputs


    
