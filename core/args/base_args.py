import argparse


class BaseArgs:
    def __init__(self, cfg):
        self.cfg = cfg
        self.parser = argparse.ArgumentParser(description='%s Train Parser' % self.cfg['MODEL'].upper())   # model name

    def _gen_fix_args(self):
        # ------------------------------------------基础设置----------------------------------------#
        # 实验名字
        self.parser.add_argument("-expn", "--experiment_name", type=str, default=self.cfg['EXPERIMENT_NAME'])
        # 模型名字
        self.parser.add_argument("-n", "--name", type=str, default=self.cfg['MODEL'], help="model name")
        # 加载已训练好的模型路径，测试的时候调用
        self.parser.add_argument("--checkpoint", type=str, default=self.cfg['CHECK_POINT'],
                                 help="path of trained model")

        # ------------------------------------------1. dataset----------------------------------------#
        self.parser.add_argument("--root_path", type=str, default=self.cfg['ROOT_PATH'], help="dataset root path")
        self.parser.add_argument('--dataset', default=self.cfg['DATASET'], type=str, help="dataset name")
        self.parser.add_argument('--device', default=self.cfg['DEVICE'], type=str, help="device for training")
        # 不可调
        self.parser.add_argument('--num_nodes', default=self.cfg['num_nodes'], type=int, help="num nodes of dataset")
        self.parser.add_argument("-l", '--length', default=self.cfg['total_length'], type=int, help="total time series length of dataset")
        self.parser.add_argument('--train_length', default=self.cfg['train_length'], type=int,
                                 help="train time series length of dataset")

        self.parser.add_argument('--map_fea_num', default=self.cfg['map_fea_num'], type=int,
                                 help="map_fea_num")
        self.parser.add_argument('--map_num', default=self.cfg['map_num'], type=int,
                                 help="map_num")
        self.parser.add_argument('--enh_fea_num', default=self.cfg['enh_fea_num'], type=int,
                                 help="enh_fea_num")
        self.parser.add_argument('--enh_num', default=self.cfg['enh_num'], type=int,
                                 help="enh_num")

        self.parser.add_argument('--channel', default=self.cfg['channel'], type=int, help="feature num of dataset")
        self.parser.add_argument('--features', default=self.cfg['features'], type=list, help="features of dataset")
        self.parser.add_argument('--mode', default=self.cfg['mode'], type=str, help=" mode of dataloader")

        # 可调
        self.parser.add_argument('--num_time_steps_in', default=self.cfg['num_time_steps_in'], type=int,
                                 help="input length of dataset")
        self.parser.add_argument('--num_time_steps_out', default=self.cfg['num_time_steps_out'], type=int,
                                 help="output length of dataset")

        self.parser.add_argument('--win_size', default=self.cfg['num_time_steps_in'], type=int,
                                 help="input length of dataset")
        self.parser.add_argument('--slide_step', default=self.cfg['slide_step'], type=int,
                                 help="slide_step of dataset")
        # 不可调
        self.parser.add_argument('--predict_feature', default=self.cfg['predict_feature'],
                                 help="predict_feature")
        self.parser.add_argument('--predict_step', default=self.cfg['predict_step'],type=int,
                                 help="predict_step")
        self.parser.add_argument('--train_size', default=self.cfg['train_size'], type=float,
                                 help="train_size")


        self.parser.add_argument('--normalizer', default=self.cfg['normalizer'], type=str,
                                 help="normalizer type of dataset")

        # ---------------------------------------2. default/other---------------------------------------#
        # 可调: little important
        self.parser.add_argument('--epochs', default=self.cfg['epochs'], type=int,help="training epochs")
        self.parser.add_argument('--patience', default=self.cfg['patience'], type=int,
                                 help="patience epoch number of early stopping when training,"
                                 "which means that the training score has not increased "
                                 "for patience consecutive epochs")
        self.parser.add_argument('--seed', default=self.cfg['seed'], type=int, help="random seed for numpy/pytorch")

        # 可调: important
        self.parser.add_argument('--batch_size', default=self.cfg['batch_size'], type=int,
                                 help="batch size of datasets when training/evaluation")
        self.parser.add_argument('--lr', default=self.cfg['lr'], type=bool,
                                 help="init learning rate of training")

        # 不需要调
        self.parser.add_argument('--num_workers', default=self.cfg['num_workers'], type=int,
                                 help="(int) number of worker threads for data loading")
        self.parser.add_argument('--resume', default=self.cfg['resume'], type=bool,
                                 help="resume training from last checkpoint")