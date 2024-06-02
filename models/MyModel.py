import torch
import torch.nn as nn
from models.GPT import GPT
from models.VITST import VITST

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.gpt = GPT(args)
        self.vitst = VITST(args)
        self.construct = nn.Linear(1080, args.win_size)

    def forward(self, raw, picture):
        B, L, C = raw.shape
        output_raw = self.gpt(raw)
        ouput_picture = self.vitst(picture)

        #计算最终重构的时序
        # output_time = output_raw + ouput_picture
        # output_time = self.construct(output_time)
        # output_time = output_time.reshape(B,C,L).transpose(1,2)

        #消融图片
        output_time = self.construct(output_raw)
        output_time = output_time.reshape(B,C,L).transpose(1,2)

        return output_raw, ouput_picture, output_time
