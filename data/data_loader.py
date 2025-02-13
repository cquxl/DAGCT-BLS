import random
import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from models import FrozenBLS
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from utils import normalize, setup_seed



setup_seed(42)


class ChaoticDataset(Dataset):
    '''
    two kinds of datasets to yield
    1. generate:one to one,which means num of input time steps and num of output time steps equal, \
    like the Table Forecast or generation prediction
    2. slide window: Sliding Window Forecast, like the traditional time series prediciton, \
    using historical time steps predict future time steps
    if use 'generate' to predict, num of input time steps and num of output time steps equal 1
    '''
    def __init__(self, win_size=100, slide_step=1,
                 predict_feature='x', predict_step=1,
                 train_length=3000, train_size=0.7,
                 map_fea_num=6, map_num=5,
                 enh_fea_num=41, enh_num=1,
                 root_path='../dataset',
                 type='lorenz', mode='train', predict_mode='generate'):

        self.file_name = os.path.join(root_path, 'multi_bls_data', type, '%s_%s' % (predict_feature, predict_step))
        self.win_size = win_size
        self.slide_step = slide_step
        self.train_length = train_length
        self.train_size = train_size
        self.type = type
        self.mode = mode
        self.predict_step = predict_step
        self.predict_mode = predict_mode
        self.fb = FrozenBLS(map_fea_num, map_num, enh_fea_num, enh_num,
                            type=type, root_path=os.path.join(root_path, 'standard_data'))
  
        if predict_mode == 'generate':
            X, Y = self.fb.split_data(predict_feature, predict_step, train_length, file_dir=os.path.join(root_path, 'multi_bls_data'),
                                      is_split=False, no_bls=False)
            # X, Y = self.fb.split_data(predict_feature, predict_step, train_length, file_dir=os.path.join(root_path, 'multi_bls_data'),
            #                           is_split=False, no_bls=True)

        elif predict_mode == 'slide window':
            X, Y = self.fb.split_data(predict_feature, 0, train_length, file_dir=os.path.join(root_path, 'multi_bls_data'),
                                      is_split=False)
        else:
            raise ValueError

  
        if self.type == 'lorenz' or self.type == 'rossler':
            self.features = ['x', 'y', 'z']
        if self.type == 'sea_clutter':
            self.features = ['feature_%s' % i for i in range(X.shape[0])]
        self.X = []
        for i in range(X.shape[0]):
            x_scaler = normalize(default='MinMaxScaler')
            x = x_scaler.fit_transform(X[i])
            self.X.append(x)
        self.X = np.array(self.X)
        self.y_scaler = normalize(default='MinMaxScaler')
        self.Y = self.y_scaler.fit_transform(Y)
   
        assert self.X.shape[1] > self.train_length
        self.data = self.X[:,:self.train_length,:]        # [3,3000,71]
        self.label = self.Y[:self.train_length,:]         # [3000,1]
        self.test = self.X[:,self.train_length:,:]   # [3,935,71]
        self.test_label = self.Y[self.train_length:,:]    # [935,1]
        # train,valid
        self.train, self.val, self.train_label, self.val_label = [], [], [], []
        for i in range(self.data.shape[0]):
            data_i = self.data[i]
            label_i = self.label
            train_x, val_x, train_y, val_y = train_test_split(data_i, label_i, train_size=self.train_size,
                                                              shuffle=True, random_state=42)
            self.train.append(train_x)
            self.val.append(val_x)
            self.train_label = train_y
            self.val_label = val_y
        self.train = np.array(self.train)    # [3,2100,71]
        self.val = np.array(self.val)  # [3,900,71]
        if predict_mode=='generate':
            print('-----train-----')
            print(f'train feature shape:{self.train.shape}')
            print(f'train label shape:{self.train_label.shape}')

            print('-----val-----')
            print(f'val feature shape:{self.val.shape}')
            print(f'val label shape:{self.val_label.shape}')

            print('-----test-----')
            print(f'test feature shape:{self.test.shape}')
            print(f'test label shape:{self.test_label.shape}')

    def __len__(self):
        if self.mode == "train":
            if self.predict_mode == 'generate':
                return (self.train.shape[1] - self.win_size) // self.slide_step + 1
            return (self.Y.shape[0]-self.win_size-self.predict_step)//self.slide_step + 1
        elif self.mode == "val":
            if self.predict_mode == 'generate':
                return (self.val.shape[1] - self.win_size) // self.slide_step + 1
            return (self.Y.shape[0] - self.win_size - self.predict_step) // self.slide_step + 1
        elif (self.mode == 'test'):
            if self.predict_mode == 'generate':
                return (self.test.shape[1] - self.win_size) // self.slide_step + 1
            return (self.Y.shape[0] - self.win_size - self.predict_step) // self.slide_step + 1
        else:
            raise ValueError

    def __getitem__(self, item):
        index = item * self.slide_step 
        if self.mode == "train": # train, train_label
            if self.predict_mode == 'generate':
                return np.float32(self.train[:, index:index+self.win_size, :]), \
                    self.train_label[index:index + self.win_size, :]
            return np.float32(self.X[:, index:index+self.win_size, :]), \
                self.Y[index+self.win_size:index+self.win_size+self.predict_step, :]
                # self.train_label[index:index+self.win_size, :]

        elif self.mode == "val":
            if self.predict_mode == 'generate':
                return np.float32(self.val[:, index:index + self.win_size, :]), \
                    self.val_label[index:index + self.win_size, :]
            return np.float32(self.X[:, index:index + self.win_size, :]), \
                self.Y[index+self.win_size:index+self.win_size+self.predict_step, :]
                # self.val_label[index:index + self.win_size, :]
        elif self.mode == "test":
            if self.predict_mode == 'generate':
                return np.float32(self.test[:, index:index + self.win_size, :]), \
                    self.test_label[index:index + self.win_size, :]
            return np.float32(self.X[:, index:index + self.win_size, :]), \
                self.Y[index+self.win_size:index+self.win_size+self.predict_step,  :]
                # self.test_label[index:index + self.win_size, :]
        else:
            raise ValueError


def get_data_loader(dataset, batch_size=100, num_workers=0, mode='train'):
    x, y = dataset[0]
    if mode == 'train':
        shuffle = True
    else:
        shuffle = False
    if dataset.predict_mode=='slide window':
        dataset1 = torch.utils.data.Subset(dataset, range(dataset.train_length))
        train_indices, val_indices = train_test_split(range(len(dataset1)), train_size=dataset.train_size, random_state=42)
        train_dataset = torch.utils.data.Subset(dataset1, train_indices)
        val_dataset = torch.utils.data.Subset(dataset1, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, range(dataset.train_length,len(dataset)))
        if mode == 'train':
            print(f'-------------{mode} dataset length:{len(train_dataset)}-------------')

            print(f'x shape:{x.shape}')
            print(f'y shape:{y.shape}')
            return DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers), dataset.y_scaler
        elif mode == 'val':
            print(f'-------------{mode} dataset length:{len(val_dataset)}-------------')
            print(f'x shape:{x.shape}')
            print(f'y shape:{y.shape}')
            return DataLoader(dataset=val_dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers), dataset.y_scaler
        elif mode == 'test':
            print(f'-------------{mode} dataset length:{len(test_dataset)}-------------')
            print(f'x shape:{x.shape}')
            print(f'y shape:{y.shape}')
            print('-------------%s-------------' % mode)
            return DataLoader(dataset=test_dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers), dataset.y_scaler
        else:
            raise ValueError
    else:
        print(f'-------------{mode} dataset length:{len(dataset)}-------------')
        print(f'x shape:{x.shape}')
        print(f'y shape:{y.shape}')
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers)
    return data_loader, dataset.y_scaler

if __name__ == '__main__':
    dataset = ChaoticDataset(win_size=1, slide_step=1,
                             predict_feature='x', predict_step=3,
                             train_length=3000, train_size=0.7,
                             root_path='../dataset/',
                             type='lorenz', mode='train', predict_mode='generate')
    train_loader, y_scaler = get_data_loader(dataset, mode='train',
                                             batch_size=64, num_workers=0) # slide window
    val_loader, _ = get_data_loader(dataset, mode='val',
                                    batch_size=64, num_workers=0) # slide window
    test_loader, _ = get_data_loader(dataset, mode='test',
                                     batch_size=64, num_workers=0) # slide window
    x, y = next(iter(train_loader))
    print(x.shape)
    print(y.shape)
