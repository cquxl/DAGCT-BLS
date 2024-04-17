# Author: Xiong Lang
# Date: 2023/09/10


import argparse
import os
import torch
from utils import read_yaml_to_dict, get_cfg, setup_seed, print_args_model_info
from loguru import logger
from tabulate import tabulate
from exp import Exp
from core import *

torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#-----------------------------------------参数文件获取---------------------------------------------#
DATASET = 'lorenz'
DEVICE = 'cuda:0'
MODEL = 'DAGCT_BLS'    #  -->改,只需要改此参数即可自动运行
ROOT_PATH = './dataset/'


# 读取基本的参数文件
cfg_data_file = 'cfg/datasets/%s.yaml' % DATASET
cfg_model_file = 'cfg/models/%s.yaml' % (MODEL.lower())
cfg_default_file = 'cfg/default.yaml'
cfg_data, cfg_model, cfg_defalut, cfg_all = get_cfg(cfg_data_file, cfg_model_file, cfg_default_file)
EXPERIMENT_NAME = '%s_%s_%s' % (MODEL, cfg_defalut['batch_size'], cfg_defalut['lr'])
CHECK_POINT = os.path.join('logs', MODEL, DATASET, EXPERIMENT_NAME, 'checkpoint.pth')
cfg_all['DATASET'] = DATASET
cfg_all['DEVICE'] = DEVICE
cfg_all['MODEL'] = MODEL
cfg_all['ROOT_PATH'] = ROOT_PATH
cfg_all['EXPERIMENT_NAME'] = EXPERIMENT_NAME
cfg_all['CHECK_POINT'] = CHECK_POINT

# 模型的参数解析
args = gen_args[MODEL](cfg_all).args

model = get_model_dict(args)[MODEL]
args.model = model
setup_seed(args.seed)
exp = Exp(args)

def main(mode='train'):
    args.mode = mode
    trainer = trainer_dict[MODEL](exp, args)
    if args.mode == 'train':
        trainer.train()
        torch.cuda.empty_cache()
    else : # test/val
        trainer.evaluate(save_pred=True, inverse=False, checkpoint=args.checkpoint)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main('train')






