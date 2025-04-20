import logging
from logging import INFO
import sys as sys
import warnings
from collections import defaultdict
import scipy.special
import torch
warnings.filterwarnings("ignore")
import numpy as np
from Dependency.Model import *


def DistAvgAdaptive(logger, current_local_models, weight_matrix, cnt_round, acc_list):
    cur_models, model_layer_list = [], current_local_models[0].keys()
    cur_models_vec, cur_weights_shape = [], {k: current_local_models[0][k].shape for k in model_layer_list}
    for i in range(len(current_local_models)):
        cur_model = torch.cat([current_local_models[i][k].reshape(-1) for k in model_layer_list])
        cur_models_vec.append(cur_model)
    cur_models_vec = torch.vstack(cur_models_vec)

    for i in range(len(current_local_models)):
        cur_adj = np.where(weight_matrix[i])[0]
        logger.log(INFO, f'DistAvg: {i} {cur_adj}')
        weights = np.ones(len(cur_adj))
        if adap_agg:
            if (cnt_round + 1) >= sample_start_round:
                for adj_i, j in enumerate(cur_adj):
                    weights[adj_i] = acc_list[j][-1]
                mean_weight = np.mean(weights)
                weights -= mean_weight
            weights = scipy.special.softmax(weights)
        elif acc_adap_agg:
            if (cnt_round + 1) > 5:
                for adj_i, j in enumerate(cur_adj):
                    weights[adj_i] = acc_list[j][-1]
                mean_weight = np.mean(weights)
                weights -= mean_weight
                print(f'Worker {i}:', weights)
            weights = scipy.special.softmax(weights)
        else:
            if not same_weights:
                for adj_i, j in enumerate(cur_adj):
                    if j in l_client_ids:
                        weights[adj_i] = 10.0
                    elif j in m_client_ids:
                        weights[adj_i] = 5.0
            weights = weights / sum(weights)

        log_weights = [f'{w:.4f}' for w in weights]
        logger.log(INFO, f"Weights: [{', '.join(log_weights)}]")
        weighted_wvec = torch.sum(torch.from_numpy(np.float32(weights))[:, None] * cur_models_vec[cur_adj], dim=0)
        offset, cur_f_model = 0, {}
        for k in model_layer_list:
            cur_shape = cur_weights_shape[k]
            shape_size = cur_shape.numel()
            cur_f_model[k] = weighted_wvec[offset:offset+shape_size].reshape(cur_shape)
            offset += shape_size
        cur_models.append(cur_f_model)
    current_local_models = cur_models
    return current_local_models
