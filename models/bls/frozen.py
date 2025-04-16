from .BLS import BLS
import numpy as np
import os
import matplotlib.pyplot as plt
import d2l.torch as d2l


class FrozenBLS:
    def __init__(self, map_fea_num=6, map_num=5, enh_fea_num=41, enh_num=1,
                 type='lorenz', root_path='../data/standard_data'):
        path = os.path.join(root_path, '%s_rec_dict.npy' % type)
        self.rec_dict = np.load(path, allow_pickle=True).tolist()
        self.map_fea_num = map_fea_num
        self.map_num = map_num
        self.enh_fea_num = enh_fea_num
        self.enh_num = enh_num
        self.bls = BLS(map_fea_num, map_num, enh_fea_num, enh_num)
        self.type = type

    def split_data(self, predict_feature='x', predict_step=1, train_length=3000,
                   file_dir='../data/multi_bls_data', is_split=True, no_bls=False):
        file_dir = os.path.join(file_dir,self.type)
        save_dir = '%s_%s' % (predict_feature, predict_step)
        if not os.path.exists(os.path.join(file_dir, save_dir)):
            os.makedirs(os.path.join(file_dir, save_dir))
        file_name = os.path.join(file_dir, save_dir)

        assert predict_feature in list(self.rec_dict.keys())
        X_train_bls = []
        X_test_bls = []
        all_X = []
        for feature, rec_data in self.rec_dict.items():
            
            x_feature = rec_data[:rec_data.shape[0]-predict_step,:]

            # if not is_split:
            if no_bls:
                
                X = np.zeros((x_feature.shape[0], 71))
                X[:,:x_feature.shape[1]] = x_feature
            else:
                X = self.bls.generate_features(x_feature, is_train=True)
            all_X.append(X)

            assert x_feature.shape[0] > train_length
            train_x = x_feature[:train_length, :]
            test_x = x_feature[train_length:, :]

            bls_train_x = self.bls.generate_features(train_x, is_train=True)
            X_train_bls.append(bls_train_x)
            bls_test_x = self.bls.generate_features(test_x, is_train=False)
            X_test_bls.append(bls_test_x)
            if feature == predict_feature:
                Y = [rec_data[:,-1][i+1:i+predict_step+1] for i in range(rec_data.shape[0]-predict_step)]
                if predict_step == 0:
                    # Y = np.array(Y).reshape(-1,1) # 全量集
                    Y = rec_data[:,-1].reshape(-1,1)
                else:
                    Y = np.array(Y).reshape(-1, predict_step)
                train_y = Y[:train_length, :] #[3000,1]
                test_y = Y[train_length:, :]
        X_train_bls = np.array(X_train_bls) #[3,3000,71]
        X_test_bls = np.array(X_test_bls) #[3,l+,71]
        all_X = np.array(all_X) #[]
        if not is_split:
            print(f'All bls X shape:{all_X.shape}')
            print(f'All y shape:{Y.shape}')
    
            np.save('%s/X_%s.npy' % (file_name,predict_step), all_X)
            np.save('%s/Y_%s.npy' % (file_name,predict_step), Y)
            return all_X, Y
        print(f'train bls X shape:{X_train_bls.shape}')
        print(f'test bls X shape:{X_test_bls.shape}')
        print(f'train Y shape:{train_y.shape}')
        print(f'test Y shape:{test_y.shape}')
    
        np.save('%s/X_train_bls.npy' % file_name, X_train_bls)
        np.save('%s/X_test_bls.npy' % file_name, X_test_bls)
        np.save('%s/train_y.npy' % file_name, train_y)
        np.save('%s/test_y.npy' % file_name, test_y)
        return X_train_bls, X_test_bls, train_y, test_y

