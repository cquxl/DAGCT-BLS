import torch
import torch.nn as nn
import torch.nn.functional as F

from .Transformer import Transformer
from .DAGCN import DAGCN


class MLPHead(nn.Module):
    def __init__(self,c_in, c_out, input_time_steps, output_time_steps, num_nodes_in, num_nodes_out):
        super(MLPHead, self).__init__()
        # self.linear1 = nn.Linear(c_in, c_out)
        self.linear1 = nn.Linear(input_time_steps, output_time_steps)
        self.linear2 = nn.Linear(c_in, c_out)
        self.linear3 = nn.Linear(num_nodes_in, num_nodes_out)
        self.activation = nn.ReLU()

    def forward(self, x): # [B,N,T,C]
        # predict future
        out1 = self.linear1(x.permute(0,1,3,2)) #
        # out1 = self.activation(out1) # [B,N,C,pred_len]

        # 预测变量值
        out2 = self.linear2(out1.permute(0,1,3,2))
        # out2 = self.activation(out2)   # [B,N,pred_len,c_out]

        # 预测节点值
        out3 = self.linear3(out2.permute(0,2,3,1))
        # out3 = self.activation(out3)   # [B,pred_len,c_out, n_out]

        # out = out3.permute(0, 3, 1, 2)  # [B,n_out,pred_len,c_out]
        # 变成[B, pred_len, 1]
        out = out3.squeeze(-1)
        return out


class DAGCT_BLS(nn.Module):
    def __init__(self, num_time_steps_in, num_time_steps_out,
                 num_nodes, in_channels, hidden_dims, cheb_k, embed_dim,
                 out_channels, seg_len, d_model, n_heads,
                 dropout, num_layers, mode='full', spatial_attention=True, temporal_attention=True
                 ):
        super(DAGCT_BLS, self).__init__()

        self.dagcn = DAGCN(num_time_steps_in, num_nodes, in_channels, hidden_dims,
                           cheb_k, embed_dim, mode, spatial_attention)

        self.transformer_encoder = Transformer(num_nodes, seg_len, in_channels=hidden_dims,
                                               d_model=d_model, n_heads=n_heads, dropout=dropout,
                                               num_layers=num_layers, output_attention=temporal_attention)

        # self.transformer_encoder1 = Transformer(num_nodes, seg_len, in_channels=num_time_steps_in,
        #                                        d_model=d_model, n_heads=n_heads, dropout=dropout,
        #                                        num_layers=num_layers, output_attention=temporal_attention)

        self.mlp_head = MLPHead(d_model, out_channels,
                                num_time_steps_in, num_time_steps_out,
                                num_nodes, num_nodes_out=1)
        # self.mlp_head1 = MLPHead(d_model, out_channels,
        #                         hidden_dims, num_time_steps_out,
        #                         num_nodes, num_nodes_out=1)

    def forward(self,x): # [B,N,L,C]
        # spatial
        x_s, spatial_attn = self.dagcn(x.permute(0,2,1,3)) # [B,L,N,C]
        # temporal
        x_t, temporal_attn = self.transformer_encoder(x_s.permute(0,2,1,3))
        # x_t, temporal_attn = self.transformer_encoder(x)
        # x_t, temporal_attn = self.transformer_encoder1(x_s.permute(0, 2, 3, 1)) #[B,N,h,d_model]

        # pred
        out = self.mlp_head(x_t)  # [B,1,pred_len,c_out]
        # out = self.mlp_head(x_s.permute(0,2,1,3))
        # out = self.mlp_head1(x_t)  # [B,1,pred_len,c_out]
        # return out, None, temporal_attn
        return out, spatial_attn, temporal_attn
        # return out, spatial_attn, None


