import os
import gc
import sys
import time
import logging
from abc import ABC, abstractmethod
from logging import Formatter, StreamHandler, FileHandler, Filter
from torch.utils.tensorboard import SummaryWriter

class LevelFilter(Filter):
    def __init__(self, name: str='', level: int=logging.INFO) -> None:
        super().__init__(name=name)
        self.level = level

    def filter(self, record):
        if record.levelno < self.level:
            return False
        return True

class BaseTrainer(ABC):
    def _initialize_logger(self, log_dir='.details/logs/'):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            print(f'Make log dir: {os.path.join(os.getcwd(), log_dir)}')
        
        self.logger = logging.getLogger('logger')
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False # 防止向上传播导致root logger也打印log
        self.logger.train_time_str = time.strftime("%m-%d_%H:%M:%S", time.localtime())

        stdf = StreamHandler(sys.stdout)
        stdf.addFilter(LevelFilter('std_filter', logging.INFO))
        stdf.setFormatter(Formatter('[%(levelname)s]: %(message)s'))
        self.logger.addHandler(stdf)

        filef = FileHandler(f'{log_dir}/log_train_{self.logger.train_time_str}.txt', 'w')
        filef.addFilter(LevelFilter('file_filter', logging.INFO))
        filef.setFormatter(Formatter('[%(levelname)s %(asctime)s] %(message)s', "%Y%m%d-%H:%M:%S"))
        self.logger.addHandler(filef)

    def __init__(self, checkpoint_root):
        self._initialize_logger()
        self.writer = SummaryWriter(log_dir=f'.details/runs/{self.logger.train_time_str}')
        self.checkpoint_dir = os.path.join('.details/checkpoints/', checkpoint_root, self.logger.train_time_str+'/')
        os.makedirs(self.checkpoint_dir)

    @abstractmethod
    def __del__(self):
        self.writer.close()
        self.logger.info('Close self.writer')
        if not os.listdir(self.checkpoint_dir):
            os.remove(self.checkpoint_dir)
            self.logger.info('Remove self.checkpoint_dir')
        gc.collect()

    @abstractmethod
    def train(self):
        pass

