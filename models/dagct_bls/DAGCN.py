import torch
import torch.nn.functional as F
import torch.nn as nn


class DAGCN_reduce(nn.Module):
    def __init__(self, num_time_steps, num_nodes, in_dims, out_dims, cheb_k, embed_dim, spatial_attention=False):
        '''
        mode:'full','reduce'
        if reduce dimension: embed_dim<N
        '''
        super(DAGCN_reduce, self).__init__()
        self.num_time_steps = num_time_steps
        self.num_nodes = num_nodes
        self.cheb_k = cheb_k
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.embed_dim = embed_dim
        self.spatial_attention = spatial_attention

       
        self.dn_embeddings = nn.Parameter(torch.randn(num_time_steps,
                                                      num_nodes,
                                                      embed_dim),
                                          requires_grad=True)    # [T, N, embed_dim]

        # Theta = E*W--->(tnd,d
        self.weights_pool = nn.Parameter(torch.randn(num_time_steps,embed_dim,
                                                           cheb_k,
                                                           in_dims,
                                                           out_dims),
                                         requires_grad=True)
        self.bias_pool = nn.Parameter(torch.randn(num_time_steps,embed_dim,
                                                        out_dims),
                                      requires_grad=True)

    def forward(self, x): # x-->[B,T,N,C]
        supports = F.softmax(F.relu(torch.einsum('tne, tse->tns', self.dn_embeddings, self.dn_embeddings)), dim=-1)
        unit = torch.stack([torch.eye(self.num_nodes).to(supports.device) for _ in range(self.num_time_steps)])
        support_set = [unit, supports]
        #default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.einsum('tnn, tns->tns', 2 * supports, support_set[-1])- support_set[-2])
        supports = torch.stack(support_set, dim=1) # [T, cheb_k, N,N]
        # supports = supports[:,:1,:,:]

        # theta
        theta = torch.einsum('tnd, tdkio->tnkio', self.dn_embeddings, self.weights_pool) #T, N, cheb_k, dim_in, dim_out
        bias = torch.einsum('tnd, tdo->tno', self.dn_embeddings, self.bias_pool)  #T, N, dim_out
        x_g = torch.einsum('tknm, btmc->btknc', supports, x)
        x_gconv = torch.einsum('btkni, tnkio->btno', x_g, theta) + bias
        if self.spatial_attention:
            return x_gconv, supports
        return x_gconv, None    # [B, T, N, dim_out]

class DAGCN_full(nn.Module):
    def __init__(self, num_time_steps, num_nodes, in_dims, out_dims, cheb_k, spatial_attention=False):

        super(DAGCN_full, self).__init__()
        self.num_time_steps = num_time_steps
        self.num_nodes = num_nodes
        self.cheb_k = cheb_k
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.spatial_attention = spatial_attention

        self.dn_embeddings = nn.Parameter(torch.randn(num_time_steps,
                                                      num_nodes,
                                                      ),
                                          requires_grad=True)    # [T, N, embed_dim]
        # Theta = E*W--->(tnd,d
        self.weights_pool = nn.Parameter(torch.randn(num_time_steps,num_nodes,cheb_k,
                                                           in_dims,
                                                           out_dims),
                                         requires_grad=True)
        self.bias_pool = nn.Parameter(torch.randn(num_time_steps,num_nodes,out_dims),
                                      requires_grad=True)

    def forward(self, x): # x-->[B,T,N,C]
        supports = F.softmax(F.relu(torch.einsum('tn, ts->tns', self.dn_embeddings, self.dn_embeddings)), dim=-1)
        unit = torch.stack([torch.eye(self.num_nodes).to(supports.device) for _ in range(self.num_time_steps)])
        support_set = [unit, supports]
        for k in range(2, self.cheb_k):
            support_set.append(torch.einsum('tnn, tns->tns', 2 * supports, support_set[-1])- support_set[-2])
        # supports = torch.stack(support_set, dim=1) # [T, cheb_k, N,N]
        # supports = supports[:,:1,:,:]
        theta = self.weights_pool
        bias = self.bias_pool
        x_g = torch.einsum('tknm, btmc->btknc', supports, x)
        x_gconv = torch.einsum('btkni, tnkio->btno', x_g, theta) + bias
        if self.spatial_attention:
            return x_gconv, supports
        return x_gconv, None

class DAGCN(nn.Module):
    def __init__(self, num_time_steps, num_nodes, in_dims, out_dims, cheb_k, embed_dim, mode, spatial_attention):
        super(DAGCN, self).__init__()
        if mode == 'full':
            self.dagcn = DAGCN_full(num_time_steps, num_nodes, in_dims, out_dims, cheb_k, spatial_attention)
        elif mode == 'reduce':
            self.dagcn = DAGCN_reduce(num_time_steps, num_nodes, in_dims, out_dims, cheb_k, embed_dim, spatial_attention)
        else:
            raise ValueError
    def forward(self, x):
        out, attn = self.dagcn(x)
        return F.relu(out), attn





