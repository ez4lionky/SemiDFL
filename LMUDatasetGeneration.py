import csv
import pandas as pd
import random as rd
from torchvision import datasets
from pathlib import Path
import math
import numpy as np
from utils.dist_wm_table import get_topology
from collections import defaultdict
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from argparse import ArgumentParser
from GlobalParameters import str2bool


def build_non_iid_by_dirichlet(
        random_state, indices2targets, non_iid_alpha, num_classes, num_indices, n_workers
):
    n_auxi_workers = 10
    # assert n_auxi_workers <= n_workers

    # random shuffle targets indices.
    random_state.shuffle(indices2targets)

    # partition indices.
    from_index = 0
    splitted_targets = []
    num_splits = math.ceil(n_workers / n_auxi_workers)
    split_n_workers = [
        n_auxi_workers
        if idx < num_splits - 1
        else n_workers - n_auxi_workers * (num_splits - 1)
        for idx in range(num_splits)
    ]
    split_ratios = [_n_workers / n_workers for _n_workers in split_n_workers]
    for idx, ratio in enumerate(split_ratios):
        to_index = from_index + int(n_auxi_workers / n_workers * num_indices)
        splitted_targets.append(
            indices2targets[
            from_index: (num_indices if idx == num_splits - 1 else to_index)
            ]
        )
        from_index = to_index

    idx_batch = []
    for _targets in splitted_targets:
        # rebuild _targets.
        _targets = np.array(_targets)
        _targets_size = len(_targets)

        # use auxi_workers for this subset targets.
        _n_workers = min(n_auxi_workers, n_workers)
        n_workers = n_workers - n_auxi_workers

        # get the corresponding idx_batch.
        min_size = 0
        # ensure the final number of samples in each client is at least half
        while min_size < int(0.50 * _targets_size / _n_workers):
            _idx_batch = [[] for _ in range(_n_workers)]
            for _class in range(num_classes):
                # get the corresponding indices in the original 'targets' list.
                idx_class = np.where(_targets[:, 1] == _class)[0]
                idx_class = _targets[idx_class, 0]
                # sampling.
                try:
                    proportions = random_state.dirichlet(
                        np.repeat(non_iid_alpha, _n_workers)
                    )
                    # balance
                    proportions = np.array(
                        [
                            p * (len(idx_j) < 1.1 * _targets_size / _n_workers)
                            for p, idx_j in zip(proportions, _idx_batch)
                        ]
                    )
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_class)).astype(int)[:-1]
                    _idx_batch = [
                        idx_j + idx.tolist()
                        for idx_j, idx in zip(
                            _idx_batch, np.split(idx_class, proportions)
                        )
                    ]
                    min_size = min([len(idx_j) for idx_j in _idx_batch])
                except ZeroDivisionError:
                    pass
        idx_batch += _idx_batch
    return idx_batch


def prepare_data(client_indices, targets, num_class, client_ids):
    num_clients = len(client_indices)
    targets_clients = [targets[cd] for cd in client_indices]
    class_num_clients = [Counter(tc) for tc in targets_clients]

    total_class_counts = Counter(np.concatenate(targets_clients))
    # data = np.zeros((num_clients, num_class), dtype=int)
    data = np.zeros((num_clients, num_class), dtype=float)
    for i in range(num_clients):
        for j in range(num_class):
            if total_class_counts[j] > 0:
                data[i, j] = class_num_clients[i][j] / total_class_counts[j]

    print('Class distribution: ')
    for i in range(num_clients):
        print(f'Client {client_ids[i]}: {sorted(class_num_clients[i].items())}')
        print(f'Client {client_ids[i]} the number of all samples: {sum(class_num_clients[i].values())}')
    df = pd.DataFrame(data)
    df = df.reset_index().melt(id_vars='index')
    df.columns = ['Client IDs', 'Class labels', 'Bubble Size']
    return df


def partition_indices(indices, targets, client_ids, save_path, mode='labeled'):
    min_bsize, max_bsize = 0, 500
    fig, axes = plt.subplots(1, len(non_iid_alphas), figsize=(5 * len(non_iid_alphas), 5), sharey='row')
    n_clients = len(client_ids)
    fig_path = str(save_path / f'{args.label_ratio:.1f}_{mode}.pdf')
    for idx, alpha in enumerate(non_iid_alphas):
        print(f'============alpha: {alpha}=============')
        csv_path = save_path / f'client_00_{args.label_ratio:.1f}_{alpha:.2f}_{mode}.csv'
        if csv_path.exists():
            list_of_indices = []
            for cidx, cid in enumerate(client_ids):
                cur_csv_path = str(save_path / f'client_{cid:02d}_{args.label_ratio:.1f}_{alpha:.2f}_{mode}.csv')
                cur_indices = pd.read_csv(cur_csv_path, header=None).squeeze()
                list_of_indices.append(cur_indices)
        else:
            indices_w_labels = np.array([(idx, target) for idx, target in enumerate(targets) if idx in indices])
            list_of_indices = build_non_iid_by_dirichlet(rs, indices_w_labels, alpha, classes_n, len(indices), n_clients)
        ax = axes[idx] if len(non_iid_alphas) > 1 else axes
        df = prepare_data(list_of_indices, targets, classes_n, client_ids)
        min_ratio, max_ratio = df['Bubble Size'].min(), df['Bubble Size'].max()
        cur_bmin, cur_bmax = max_bsize * np.power(min_ratio, 0.6), max_bsize * np.power(max_ratio, 0.6)

        class_labels = list(range(classes_n))
        sns.scatterplot(data=df, x='Client IDs', y='Class labels', size='Bubble Size', sizes=(cur_bmin, cur_bmax),
                        legend=False, ax=ax, color=color)

        ax.set_xticks(range(len(client_ids)))
        ax.set_yticks(class_labels)
        ax.set_xticklabels(client_ids, fontsize=24)
        ax.set_yticklabels(class_labels, fontsize=24)
        ax.set_title(f'alpha={alpha}', fontsize=24)
        ax.xaxis.label.set_size(24)
        ax.yaxis.label.set_size(24)
        if not csv_path.exists():
            for cidx, cid in enumerate(client_ids):
                cur_csv_path = str(save_path / f'client_{cid:02d}_{args.label_ratio:.1f}_{alpha:.2f}_{mode}.csv')
                with open(cur_csv_path, 'w', encoding='utf-8', newline='') as w:
                    writer = csv.writer(w)
                    writer.writerows(np.array(list_of_indices[cidx])[None])
    handles = [
        plt.scatter([], [], s=max_bsize / 10, color=color, label='10%'),
        plt.scatter([], [], s=max_bsize / 4, color=color, label='25%'),
        plt.scatter([], [], s=max_bsize / 2, color=color, label='50%'),
        plt.scatter([], [], s=max_bsize, color=color, alpha=1, label='100%')
    ]
    fig.legend(handles=handles, loc='upper right', title='Bubble size', labelspacing=1.5, borderpad=1.5,
               bbox_to_anchor=(1., 0.935))
    plt.tight_layout(rect=[0, 0, 0.94, 1])
    # plt.show()
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close()
    return


parser = ArgumentParser(description='DFLSemi')
parser.add_argument('--task', type=str, default='fashion', help='mnist, fashion or cifar10',
                    choices=("mnist", "fashion", "cifar10"))
parser.add_argument('--label_ratio', type=float, default=0.5, help='Ratio of all labeled data, in percentage')
parser.add_argument('--dist_wm_id', type=int, default=1, help='ID of dist weight matrix')
parser.add_argument('--fully_supervised', type=str2bool, const=True, nargs='?', default=False,
                    help='Whether is fully supervised learning, i.e. upper bound')
args = parser.parse_args()
task = args.task
label_ratio = args.label_ratio / 100
dist_wm_id = args.dist_wm_id
fully_supervised = args.fully_supervised
num_client, l_client_ids, m_client_ids, u_client_ids, dist_wm = get_topology(dist_wm_id, fully_supervised)
non_iid_alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
print(f'task: {task}, label_ratio: {label_ratio} dist_wm_id: {dist_wm_id}')
classes_n = 10

# load original datasets
save_path = Path(f'./DFLSemi_data/dataset_{task}/{dist_wm_id:02d}')
save_path.mkdir(exist_ok=True, parents=True)
if task == 'mnist':
    dataset_train = datasets.MNIST('./samples/mnist', train=True, download=True)
    label_train = np.array(dataset_train.targets)
elif task == 'fashion':
    dataset_train = datasets.FashionMNIST('./samples/fashion', train=True, download=True)
    label_train = np.array(dataset_train.targets)
elif task == 'cifar10':
    dataset_train = datasets.CIFAR10('./samples/cifar10', train=True, download=True)
    label_train = np.array(dataset_train.targets)

# general settings
rs = np.random.RandomState(7)
sns.set(style="darkgrid")
color = 'red'

# shuffle indices and divided into labeled and unlabeled
data_length = len(label_train)
all_indices = np.arange(data_length)
np.random.shuffle(all_indices)
labeled_length = int(label_ratio * data_length)
unlabel_length = data_length - labeled_length
labeled_indices, unlabeled_indices = all_indices[:labeled_length], all_indices[labeled_length:]
print(f'labeled_length: {labeled_length}, unlabel_length: {unlabel_length}')

partition_indices(labeled_indices, label_train, l_client_ids + m_client_ids, save_path, 'labeled')
if unlabel_length != 0:
    partition_indices(unlabeled_indices, label_train, m_client_ids + u_client_ids, save_path, 'unlabeled')
