import subprocess
import argparse

parser = argparse.ArgumentParser(description='run the task by specifying parameters')
parser.add_argument('--task', type=str, choices=['mnist', 'cifar10', 'fashion'], default='mnist',
                    help='please specify the dataset name (mnist, cifar10, fashion)')
args = parser.parse_args()

task_configs = {
    'cifar10': {
        'model': 'resnet18',
        'epoch_iteration_num': '50',
        'label_ratio': [5, 1],
        'non_iid': [0.1, 100],
        'dist_wm_id': '1'
    },
    'mnist': {
        'model': 'cnn',
        'epoch_iteration_num': '25',
        'label_ratio': [0.1, 0.5],
        'non_iid': [100, 0.1],
        'dist_wm_id': '1'
    },
    'fashion': {
        'model': 'cnn',
        'epoch_iteration_num': '50',
        'label_ratio': [0.1, 0.5],
        'non_iid': [100, 0.1],
        'dist_wm_id': '1'
    }
}

task = args.task
config = task_configs[task]
model = config['model']
epoch_iteration_num = config['epoch_iteration_num']
label_ratio = config['label_ratio']
non_iid = config['non_iid']
dist_wm_id = config['dist_wm_id']

for ratio in label_ratio:
    for iid in non_iid:
        cmd = [
            'python',
            'DistSys.py',
            '--model',
            model,
            '--task',
            task,
            '--ssl_method',
            'DFLSemi',
            '--epoch_iteration_num',
            epoch_iteration_num,
            '--neighbor_pl',
            'True',
            '--label_ratio',
            f'{ratio}',
            '--non_iid_alpha',
            f'{iid}',
            '--dist_wm_id',
            dist_wm_id,
            '--pl_ablation',
            'none',
            '--log_dir',
            'logs',
        ]
        subprocess.run(cmd)
