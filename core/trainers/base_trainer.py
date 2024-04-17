from abc import ABCMeta, abstractmethod

class BaseTrainer(metaclass=ABCMeta):
    """
    Base class for training
    抽象方法
    1.
    2.
    """
    def __init__(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def before_train(self):
        pass

    @abstractmethod
    def train_in_epochs(self):
        pass

    @abstractmethod
    def train_one_epoch(self, epoch):
        pass

    @abstractmethod
    def vali_one_epoch(self, data_loader):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def after_train(self):
        pass





