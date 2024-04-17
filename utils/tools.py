import numpy as np
import torch
import json
import yaml
import json
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import random
from tabulate import tabulate
from loguru import logger


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return super().default(obj)

def normalize(default='MinMaxScaler'):
    if default == "StandardScaler":
        return StandardScaler()
    return MinMaxScaler(feature_range=(0, 1))


def save_dict_to_yaml(dict_value: dict, save_path: str):
    """dict保存为yaml"""
    with open(save_path, 'w', encoding='utf-8') as file:
        file.write(yaml.dump(dict_value, allow_unicode=True))
def save_dict_to_json(dict_value: dict, save_path: str):
    with open(save_path, 'w', encoding='utf-8') as file:
        json.dump(dict_value, file, ensure_ascii=False)

def read_yaml_to_dict(yaml_path: str):
    with open(yaml_path, 'r', encoding='utf-8') as file:
        dict_value = yaml.safe_load(file)
        return dict_value

def get_cfg(cfg_data_file, cfg_model_file, cfg_default_file):
    cfg_data = read_yaml_to_dict(cfg_data_file)
    cfg_model = read_yaml_to_dict(cfg_model_file)
    cfg_default = read_yaml_to_dict(cfg_default_file)
    cfg = {**cfg_data, **cfg_model, **cfg_default}
    return cfg_data, cfg_model, cfg_default, cfg

def print_args_model_info(args, model, print_model=False):
    params = vars(args)
    # 使用tabulate函数将参数表格化
    params_table = tabulate(params.items(), headers=["Parameter", "Value"], tablefmt="grid")
    # 使用Loguru的logger打印参数表格
    logger.info("\n" + params_table)
    # 获取模型的层结构
    model_architecture = []
    for name, module in model.named_children():
        model_architecture.append((name, module))
    # 使用tabulate函数将模型架构和参数表格化
    architecture_table = tabulate(model_architecture, headers=["Layer Name", "Layer"], tablefmt="grid")
    if print_model:
        logger.info("\n" + "Model Architecture:\n" + architecture_table)

def mse(test_y,y_pred):
    error = test_y-y_pred
    return np.mean(np.array(error)**2)
def rmse(test_y,y_pred):
    error = test_y-y_pred
    return np.sqrt(np.mean(np.array(error)**2))
def mae(test_y,y_pred):
    error = np.abs(np.array(test_y-y_pred))
    return np.mean(error)
def mape(test_y,y_pred):
    error = np.abs(np.array((test_y-y_pred)/(test_y+1e-6)))
    return np.mean(error)
def r_2(test_y,y_pred):
    return r2_score(test_y,y_pred)
def get_all_result(test_y, y_pred, multiple=False):
    test_y = np.nan_to_num(test_y)
    y_pred = np.nan_to_num(y_pred)
    mse_day = mse(test_y,y_pred)
    rmse_day = rmse(test_y,y_pred)
    mae_day = mae(test_y,y_pred)
    mape_day = mape(test_y,y_pred)
    r2_day = r_2(test_y,y_pred)
    if multiple:
        # print(f'mse:{mse_day}, rmse:{rmse_day}, mae:{mae_day},mape:{mape_day}
        return mse_day, rmse_day, mae_day, mape_day, None
    else:
        # print(f'mse:{mse_day}, rmse:{rmse_day}, mae:{mae_day},mape:{mape_day},r2:{r2_day}')
        return mse_day, rmse_day, mae_day, mape_day, r2_day

def re_normalization(x, _mean, _std, _min, _max, scale_type='standard'):
    if scale_type == 'standard':
        x = x*_std + _mean
        return x
    else:
        x = x * (_max-_min)+_min
        return x


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        self.val_loss_min = val_loss


