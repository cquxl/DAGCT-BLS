import torch
import torch.nn as nn
import math
from einops import rearrange, repeat


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x): # [64, 307, 12, 3]
        return self.pe[:, :x.size(-2)] # [12,512]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        # self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
        #                            kernel_size=3, padding=padding, padding_mode='circular', bias=False)# [d_model,l]
        self.tokenConv = nn.Conv2d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular',
                                   bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x): # [64, 307, 12, 3]
        # x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        x = x.permute(0, 3, 2, 1) # [64, 3, 12, 307] # 307是高，宽为12
        x = self.tokenConv(x).permute(0, 3, 2, 1) # [64, 307, 12, 512]
        return x

class PSWEmbedding(nn.Module):
    def __init__(self, seg_len, c_in, d_model=512, dropout=0.1):
        super(PSWEmbedding, self).__init__()
        self.seg_len = seg_len
        self.c_in = c_in
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.linear = nn.Linear(c_in, d_model)

    def forward(self, x): # [B, D, L, F]
        '''
        如果L能够整除L_seg-->seg_len分割有效保持不变
        如果L不能够整除L_seg-->seg_len变成全量集，不需要分割，直接全量值映射
        '''
        B, D, L, F= x.shape
        try:
            assert L % self.seg_len == 0, 'L cannot divide L_seg integrally'   # L能整除seg_len
            x_seg = rearrange(x, 'b d (seg_num seg_l) f -> b d seg_num seg_l f', seg_l=self.seg_len)
            x_embed = self.linear(x_seg) # [b d seg_num seg_l d_model]
            x_pos_embed = self.position_embedding(x_seg).unsqueeze(0).unsqueeze(0) # [1,1,1,seg_l, d_model]
            x_embed += x_pos_embed
            x_embed = rearrange(x_embed, 'b d seg_num seg_l d_model -> b d (seg_num seg_l) d_model', seg_l=self.seg_len)
        except AssertionError as e:
            # print('AssertionError:', e.__str__(), '\n-->use the full segment embedding')
            # 全量集编码
            # print('现在采用全量集编码')
            x_embed = self.linear(x)
            x_pos_embed = self.position_embedding(x)
            x_embed += x_pos_embed
        return self.dropout(x_embed)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model=512, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x): # [64, 307, 12, 3]
        value_emb = self.value_embedding(x) # [64, 307, 12, 512]
        pos_emb = self.position_embedding(x) # [12, 512]
        return self.dropout(value_emb+pos_emb)


