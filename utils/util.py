import os
import yaml
import torch
import random
import numpy as np
import scipy.io as sio
from pathlib import Path
from datetime import datetime
import logging
import matplotlib.pyplot as plt


def sm_array(shared_mem, shape, dtype=np.double):
    # create a new numpy array that uses the shared memory
    return np.ndarray(shape, dtype=dtype, buffer=shared_mem.buf)


def get_logger(config):
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    file_name_time = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    ssl_method, non_iid_alpha, label_ratio = config.ssl_method, config.non_iid_alpha, config.label_ratio
    file_name = f"{config.log_dir}/{ssl_method}_{label_ratio*100:.1f}_{non_iid_alpha}_{file_name_time}"

    if not config.debug:
        fh = logging.FileHandler(file_name + '.log')
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    logger.addHandler(sh)
    return file_name_time, logger


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

def calc_remain_time(max_iter, cur_i, iter_time):
    remain_iter = max_iter - cur_i
    remain_time = remain_iter * iter_time.avg
    t_m, t_s = divmod(remain_time, 60)
    t_h, t_m = divmod(t_m, 60)
    remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
    return remain_time


def plot_img(img, cmap=None, vmin=0, vmax=1, save_path=None):
    sizes = np.shape(img)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(height / 100, width / 100, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(str(save_path))
    plt.close()
    return
