import os
import torch
import numpy as np
from abc import ABCMeta, abstractmethod
import pprint
from tabulate import tabulate
import random


class BaseExp(metaclass=ABCMeta):
    """Base class for experiment
    * seed
    * device
    * optimizer
    * lr_scheduler
    * evaluator
    """
    def __init__(self):
        self.setup_seed()   
        self.output_dir = './logs'

    @abstractmethod
    def setup_seed(self, seed=42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    @abstractmethod
    def get_model(self):
        pass

    # @abstractmethod
    # def get_weight(self):
    #     pass
    @abstractmethod
    def get_dataset(self, flag: str):
        pass

    @abstractmethod
    def get_dataloader(self, flag):
        pass

    @abstractmethod
    def get_device(self):
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')


    @abstractmethod
    def get_criterion(self):
        pass

    @abstractmethod
    def get_optimizer(self):
        pass

    @abstractmethod
    def get_lr_scheduler(self):
        pass

    @abstractmethod
    def get_eval_dataset(self):
        pass

    @abstractmethod
    def get_eval_loader(self):
        pass


    def __repr__(self):
        table_header = ["keys", "values"]
        exp_table = [(str(k), pprint.pformat(v)) for k, v in vars(self).items() if not k.startswith("_")]
        return tabulate(exp_table, headers=table_header,  tablefmt="fancy_grid")
