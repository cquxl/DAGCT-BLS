import argparse
from core.args.base_args import BaseArgs
import os
from utils import read_yaml_to_dict, get_cfg, setup_seed, print_args_model_info

class DAGCT_BLS_GenArgs(BaseArgs):
    def __init__(self, cfg):
        super(DAGCT_BLS_GenArgs, self).__init__(cfg)
        self.cfg = cfg
        self._gen_fix_args()
        self._gen_model_args()
        self.args = self.parser.parse_args()


    def _gen_model_args(self):
        # -----------------------------------------3. model----------------------------------------#
        self.parser.add_argument('--predict_mode', default=self.cfg['predict_mode'],
                                 help="predict_mode")  # generate or slide window

        self.parser.add_argument('--in_channels', default=self.cfg['in_channels'], type=int,
                                 help="input dimension")
        self.parser.add_argument('--out_channels', default=self.cfg['out_channels'], type=int,
                                 help="output dimension")

        self.parser.add_argument('--cheb_k', default=self.cfg['cheb_k'], type=int,
                                 help="number of terms in Chebyshev polynomials")
        self.parser.add_argument('--embed_dim', default=self.cfg['embed_dim'], type=int,
                                 help="embedding dimension")
        self.parser.add_argument('--hidden_dims', default=self.cfg['hidden_dims'], type=int,
                                 help="hidden_dims")
        self.parser.add_argument("-sam", '--spatial_attention_mode', default=self.cfg['spatial_attention_mode'],
                                 help="spatial_attention_mode-->reduce or full")

        self.parser.add_argument('--seg_len', default=self.cfg['seg_len'], type=int,
                                 help="seg_len")
        self.parser.add_argument('--d_model', default=self.cfg['d_model'], type=int,
                                 help="d_model")
        self.parser.add_argument('--n_heads', default=self.cfg['n_heads'], type=int,
                                 help="n_heads")
        self.parser.add_argument('--dropout', default=self.cfg['dropout'], type=int,
                                 help="dropout")
        self.parser.add_argument('--num_layers', default=self.cfg['num_layers'], type=int,
                                 help="num of layers")

        self.parser.add_argument('--spatial_attention', default=self.cfg['spatial_attention'], type=bool,
                                 help="whether to output spatial attention")
        self.parser.add_argument('--temporal_attention', default=self.cfg['temporal_attention'], type=bool,
                                 help="whether to output temporal attention")



