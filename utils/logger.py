import os
import sys
from loguru import logger


def setup_logger(mode, save_dir):
    filename = '%s_log.log' % mode
    save_file = os.path.join(save_dir, filename)
    if os.path.exists(save_file):
        with open(save_file, "w") as log_file:
            log_file.truncate()
    logger.remove()
    logger.add(save_file, rotation="10 MB")
    logger.add(sys.stdout, colorize=True,
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
                      "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")  # 将日志输出到终端
    logger.info('This is a %s log' % mode)
    return logger




