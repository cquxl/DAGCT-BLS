from models import *
from utils import read_yaml_to_dict, get_cfg, setup_seed, print_args_model_info

models = ['DAGCT_BLS']

def get_model(model, args):
    if model == 'DAGCT_BLS':
        try:
            return DAGCT_BLS(args.num_time_steps_in, args.num_time_steps_out,
                             args.num_nodes, args.in_channels, args.hidden_dims, args.cheb_k, args.embed_dim,
                             args.out_channels, args.seg_len, args.d_model, args.n_heads,
                             args.dropout, args.num_layers, args.spatial_attention_mode, args.spatial_attention, args.temporal_attention)
        except:
            return
    else:
        raise ValueError

def get_model_dict(args):
    model_dict = {}
    for model in models:
        model_dict[model] = get_model(model, args)
    return model_dict


