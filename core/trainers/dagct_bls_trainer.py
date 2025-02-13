import torch
import numpy as np
import torch.nn as nn
import argparse
import json
import datetime
import os
import time
from loguru import logger
import glob
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from exp import Exp      
from utils import setup_logger, EarlyStopping, get_all_result, JSONEncoder
import sys
from .base_trainer import BaseTrainer



class DAGCT_BLS_Trainer(BaseTrainer):

    def __init__(self, exp, args):
        super().__init__()
        self.exp = exp
        self.args = args
        self.val_best_loss = np.inf
        self.device = self.exp.get_device()
        # self.device = torch.device('cuda')
        # self.use_model_ema =True
 
        self.grad_clip = True
        self.real_value = False
        # self.in_len = self.args.in_len
        # self.out_len = self.args.out_len
        self.num_time_steps_in = self.args.num_time_steps_in
        self.num_time_steps_out = self.args.num_time_steps_out
        self.in_len = self.args.num_time_steps_in
        self.out_len = self.args.num_time_steps_out
        self.log_interval = 10
        self.patience = 100
      
        self.train_time = 0
        self.inference_time = 0
    
        self.total_params = 0

        self.file_name = os.path.join(exp.output_dir, args.name, args.dataset, args.experiment_name)  
        self.logger = setup_logger(self.args.mode, self.file_name)
        logger.remove()
        self.logger.level("INFO")

    def train(self):
        self.before_train()
        try:
            self.train_in_epochs()
        except Exception as e:
            self.logger.error(f"An exception occurred: {e}")
        finally:
            self.after_train()

    def before_train(self):

        # self.logger.info(f'args:{self.args}')  
        # self.logger.info(f'exp value: \n{self.exp}')

        # model init
        self.model = self.exp.get_model()
        self.total_params = sum(p.numel() for p in self.model.parameters())
        # self.logger.info(f'Model Summary:{self.model}')
        # self.logger.info("Model Total Prameters:%.2fM" % (self.total_params / 1e6))
        self.optimizer = self.exp.get_optimizer()
        self.scheduler = self.exp.get_lr_scheduler()
        self.criterion = self.exp.get_criterion()
        self.train_loader = self.exp.get_dataloader(flag='train')
        # self.s_w = torch.as_tensor(self.exp.get_weight()).to(self.args.device)
        self.val_loader = self.exp.get_dataloader(flag='val')
        self.test_loader = self.exp.get_dataloader(flag='test')
        self.tb_logger = self.delete_and_create_tb_logger()
        # self.logger.remove()
        self.logger.add(os.path.join(self.file_name, '%s_log.log' % self.args.mode), rotation="10 MB")
        self.logger.add(sys.stdout, colorize=True,
                        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> "
                               "| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                               "<level>{message}</level>")
        self.logger.info(f'args:{self.args}') 
        self.logger.info(f'exp value: \n{self.exp}')
        self.logger.info(f'Model Summary:{self.model}')
        self.logger.info("Model Total Prameters:%.2fM" % (self.total_params / 1e6))
        self.logger.info('Trainging start......')

    def delete_and_create_tb_logger(self):
        tf_dir = os.path.join(self.file_name, "tensorboard")
        log_file = os.path.join(tf_dir, "events.out.tfevents.*")
        files = glob.glob(log_file)
        for file in files:
            os.remove(file)
        self.tb_logger = SummaryWriter(tf_dir)
        return self.tb_logger

    def train_in_epochs(self):

        self.early_stopping = EarlyStopping(self.patience, verbose=True, delta=0)
        count = 0
        for epoch in range(self.args.epochs):
            self.train_one_epoch(epoch)
            count += 1
            if self.early_stopping.early_stop:
                self.val_best_loss = self.early_stopping.val_loss_min
                self.logger.info('Early stopping')
                break
        if count != 0:
            self.run_time = self.train_time / count
            self.logger.info(f'model run time: {self.run_time}')
        else:
            self.run_time = 0
        self.total_params = self.total_params / 1e6
        train_metrics = {'totoal_params': self.total_params,
                         'train time': self.run_time}

        with open(os.path.join(self.file_name, 'train_metrics.json'), 'w') as f:
            json.dump(train_metrics, f)

    def train_one_epoch(self, epoch):
        self.logger.info(f'epoch {epoch} start training')
        total_loss = 0
        epoch_loss = 0
        start_time = time.time()
        epoch_time = time.time()
        for batch_idx, (x, y) in enumerate(self.train_loader):
            x = x.float().to(self.device)  # [64, 14, 50, 71]
            y = y.float().to(self.device)  # [64,l,1]
            st_outputs, s_out, t_out = self.model(x)
            self.optimizer.zero_grad()
            loss = self.criterion(st_outputs, y)
            loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            loss1, _, _ = self.compute_order_loss(st_outputs, y)

            total_loss += loss1.item()
            epoch_loss += loss1.item()
            if (batch_idx + 1) % self.log_interval == 0 and batch_idx > 0:
                cur_loss = total_loss / self.log_interval
                elapsed_time = time.time() - start_time
                self.logger.info(f"| epoch {epoch:3d} | {batch_idx + 1:5d}/{len(self.train_loader):5d} batches | "
                                 f"lr {self.exp.scheduler.get_last_lr()[0]:02.9f} |"
                                 f"iter time {elapsed_time / self.log_interval:5.2f} s | loss {cur_loss:5.5f}")
                total_loss = 0
                start_time = time.time()

        each_epoch_time = time.time() - epoch_time
        self.train_time += each_epoch_time
        self.logger.info(f" Epoch:{epoch} training end, cost time: {each_epoch_time} s")
  
        train_loss = epoch_loss / len(self.train_loader)
        val_loss = self.vali_one_epoch(self.val_loader)
        test_loss = self.vali_one_epoch(self.test_loader)


        self.tb_logger.add_scalar('train_loss', train_loss, epoch)
        self.tb_logger.add_scalar('val_loss', val_loss, epoch)
        self.tb_logger.add_scalar('test_loss', test_loss, epoch)

        self.logger.info(f'Epoch: {epoch + 1}, Steps:{len(self.train_loader)} | '
                         f'Train Loss:{train_loss:.10f} | Val Loss:{val_loss:.10f} | Test Loss:{test_loss:.10f}')
        self.early_stopping(val_loss, self.model, self.file_name)
        # self.early_stopping(test_loss, self.model, self.file_name)
        self.val_best_loss = self.early_stopping.val_loss_min
        self.scheduler.step()

    def vali_one_epoch(self, data_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(data_loader):
                x = x.float().to(self.device)  # [64, 12, 307, 3]
                y = y.float().to(self.device)  # [64,3,307]
                # st_outputs, s_attns, t_attns = self.model(self.s_w, x)
                st_outputs, s_out, t_out = self.model(x)
                loss1, _, _ = self.compute_order_loss(st_outputs, y)
                total_loss += loss1.item()
        self.model.train()  
        return total_loss / len(self.val_loader)

    def compute_order_loss(self, outputs, y):

        output1 = outputs[0,:,:].reshape(-1,1)  # [3,307]
        target1 = y[0,:,:].reshape(-1,1)
        for i in range(outputs.shape[0]):
            if i == 0:
                continue
            else:
                output1 = torch.cat((output1,outputs[i, -1,-1].reshape(-1,1)),dim=0)  # [1, 307]
                target1 = torch.cat((target1,y[i, -1,-1].reshape(-1,1)),dim=0)

        loss = self.criterion(output1, target1)
        return loss, output1.cpu().detach(), target1.cpu().detach()

    def evaluate(self, save_pred=False, inverse=False, checkpoint=None):
        logger.remove()
        logger.add(os.path.join(self.file_name, '%s_log.log' % self.args.mode), rotation="10 MB", level="INFO")
        logger.add(sys.stdout, colorize=True,
                   format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> "
                          "| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                          "<level>{message}</level>")
        self.model = self.exp.get_model()
        if checkpoint is not None and os.path.exists(checkpoint):
            model_dict = torch.load(self.args.checkpoint, map_location='cpu')
            self.model.load_state_dict(model_dict)
        else:  
            error_message = f"checkpoint file not found: {checkpoint} or checkpoint is None"
            logger.error(error_message)
            raise FileNotFoundError(error_message)
        test_loader = self.exp.get_dataloader(flag=self.args.mode)
        y_scaler = self.exp.y_scaler
        # self.s_w = torch.as_tensor(self.exp.get_weight()).to(self.args.device)
        self.criterion = self.exp.get_criterion()
        self.model.eval()

        e_output = []
        e_target = []
        evaluate_time = 0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(test_loader):
                start = time.time()
                x = x.float().to(self.device)  # [64, 12, 307, 3]
                y = y.float().to(self.device)  # [64,3,307]
                st_outputs, s_out, t_out = self.model(x)
                batch_time = time.time() - start
                evaluate_time += batch_time
                if self.real_value:  
              
                    y_batch = []
                    for i in range(y.shape[0]):
                        y_b = self.exp.y_scaler.inverse_transform(y[i].cpu().detach())  
                        y_batch.append(y_b)
                    y_batch = torch.as_tensor(np.stack(y_batch))
                  
                    _, output, target = self.compute_order_loss(st_outputs.cpu().detach(), y_batch) 
                    if batch_idx == 0:
                        e_output = output
                        e_target = target
                    else:
                        
                        e_output = torch.cat((e_output, output[self.out_len - 1, :]), dim=0)  # [l,307
                        e_target = torch.cat((e_target, target[self.out_len - 1, :]), dim=0)
                else:  
                    e_output, e_target = self.process_batch(e_output, e_target, st_outputs, y, batch_idx,
                                                            inverse=inverse)
       
        # mse, rmse, mae, mape, r2 = get_all_result(e_output.numpy(), e_target.numpy(), multiple=True)
        self.inference_time = evaluate_time / len(test_loader)
        mse, rmse, mae, mape, r2 = get_all_result(e_output.numpy().reshape(-1, 1), e_target.numpy().reshape(-1, 1),
                                                  multiple=False)

        # logger.add(os.path.join(self.file_name, '%s_log.log' % self.args.mode), rotation="10 MB", level="INFO")
        # logger.add(sys.stdout, colorize=True,
        #            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> "
        #                   "| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        #                   "<level>{message}</level>")
        metrics = {'mse': mse, 'rmse': rmse, 'mae': mae, 'mape': mape, 'r2': r2, 'inference_time': self.inference_time}
        logger.info(
            f"evaluate result-->mse:{mse:.8f} | rmse:{rmse:.8f} | mae: {mae:.8f} | mape: {mape:.8f} | r2: {r2:.8f} | inference_time: {self.inference_time:.4}s")
        true_pred_dict = {'truth': e_target.numpy(), 'pred': e_output.numpy()}

      
        # if not os.path.exists(os.path.join(self.file_name, 'evaluate_metrics.json')):
        with open(os.path.join(self.file_name, 'evaluate_metrics.json'), 'w') as f:
            json.dump(metrics, f, cls=JSONEncoder)
        if save_pred:
            
            np.save(os.path.join(self.file_name, 'true_pred_dict.npy'), true_pred_dict)
        return metrics

    def process_batch(self, e_output, e_target, outputs, y, batch_idx, inverse=False):

        if inverse:
            y_batch = []
            output_batch = []
            for i in range(y.shape[0]):
                y_b = self.exp.y_scaler.inverse_transform(y[i].cpu().detach())
                output_b = self.exp.y_scaler.inverse_transform(outputs[i].cpu().detach())
                y_batch.append(y_b)
                output_batch.append(output_b)
            y_batch = torch.as_tensor(np.stack(y_batch))
            output_batch = torch.as_tensor(np.stack(output_batch))
           
            _, output, target = self.compute_order_loss(output_batch, y_batch)  # 返回单个batch的值
            if batch_idx == 0:
                e_output = output
                e_target = target
            else:
               
                e_output = torch.cat((e_output, output[self.out_len - 1:, :]), dim=0)  # [l,307
                e_target = torch.cat((e_target, target[self.out_len - 1:, :]), dim=0)
        else:
            # e_output = outputs[0, :, :].reshape(-1, 1)  # [100,1]
            # e_target = y[0, :, :].reshape(-1, 1)  # [100,1
            # for i in range(outputs.shape[0]):
            #     if i == 0:
            #         continue
            #     else:
            #         e_output = torch.cat((e_output, outputs[i, -1, -1].reshape(-1, 1)), dim=0)
            #         e_output = torch.cat((e_target, y[i, -1, -1].reshape(-1, 1)), dim=0)
            # loss = criterion(output1, target1)
            if batch_idx == 0:
                e_output = outputs[0, :, :].reshape(-1,1).cpu()
                e_target = y[0, :, :].reshape(-1, 1).cpu()
            else:
                for j in range(outputs.shape[0]):
                    e_output = torch.cat((e_output, outputs[j, -1, -1].reshape(-1, 1).cpu()))
                    e_target = torch.cat((e_target, y[j, -1, -1].reshape(-1, 1).cpu()))
        return e_output, e_target

    def after_train(self):
        self.logger.info(f'training is done, best val loss:{self.val_best_loss}')



