import logging
from logging import INFO
from logging.handlers import QueueHandler
import multiprocessing

import torch.distributed
import torch.multiprocessing as processing
from multiprocessing import shared_memory
# from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
import warnings
from collections import deque

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')

import utils.util as u
from utils.global_config import init_global_config
from GlobalParameters import *
from Dependency.Aggregation import DistAvgAdaptive
from Dependency.Model import NeuralNetwork
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import copy
import os
import time
import sys
import math
from pathlib import Path


def Logging(queue, logger_func, cfg):
    log_time, _ = logger_func(cfg)
    queue.put(log_time)
    while True:
        try:
            record = queue.get()
            if record is None:
                break
            cur_logger = logging.getLogger(record.name)
            cur_logger.handle(record)
        except Exception as e:
            print(e)


def create_shared_memory(network):
    shared_mems, buf_names, buf_shapes, buf_dtypes = {}, {}, {}, {}
    for k in network.state_dict().keys():
        for w in range(num_worker):
            weight = test_model.state_dict()[k].cpu().numpy()
            name = f'{k}_{w}'
            # do not use specified name to avoid the same name across parallel programs
            sm = shared_memory.SharedMemory(create=True, size=weight.nbytes)
            shared_mems[name] = sm
            buf_names[name] = sm.name
            buf_shapes[name] = weight.shape
            buf_dtypes[name] = weight.dtype
    return shared_mems, (buf_names, buf_shapes, buf_dtypes)


def close_shared_memory(shared_mems):
    for k in shared_mems.keys():
        shared_mems[k].close()
        shared_mems[k].unlink()
    return


def upload_data_to_csm(shared_mems, wid, data_shapes, data_dtypes, state_dict):
    # upload data (state dict) to shared memory (i.e. update state dict to child process)
    for k in state_dict.keys():
        shared_dk = f'{k}_{wid}'
        shared_data = u.sm_array(shared_mems[shared_dk], data_shapes[shared_dk], data_dtypes[shared_dk])
        shared_data[...] = state_dict[k]
    return


def load_data_from_csm(shared_mems, wid, shapes, dtypes, s_ks):
    cur_sd = {}
    for k in s_ks:
        shared_dk = f'{k}_{wid}'
        cur_sd[k] = torch.from_numpy(u.sm_array(shared_mems[shared_dk], shapes[shared_dk], dtypes[shared_dk]))
    return cur_sd


def LocalTraining(worker_id: int, init_params, shm_info, ftime, p2c_link, c2p_link, log_que, device='cuda:0'):
    def start_sm():
        shared_mems = {}
        for dk in data_names.keys():
            cur_wid = int(dk.split('_')[-1])
            if cur_wid in cur_adj:
                sm = SharedMemory(name=data_names[dk])
                shared_mems[dk] = sm
        return shared_mems

    def close_sm(shared_mems):
        for sm in shared_mems.values():
            sm.close()
        return

    def load_data_from_psm(state_dict, shared_mems, adj_wid):
        cur_sd, adj_sd = {}, {}
        # load data (state dict) from shared memory (i.e. the aggregated state dict updated by parent process)
        for k in state_dict.keys():
            shared_dk = f'{k}_{worker_id}'
            cur_sd[k] = torch.from_numpy(u.sm_array(shared_mems[shared_dk], data_shapes[shared_dk],
                                                    data_dtypes[shared_dk])).to(device)

            adj_dk = f'{k}_{adj_wid}'
            adj_sd[k] = torch.from_numpy(u.sm_array(shared_mems[adj_dk], data_shapes[adj_dk], data_dtypes[adj_dk])).to(
                device)
        return cur_sd, adj_sd

    def upload_data_to_psm(state_dict, shared_mems):
        # upload data (state dict, loss) to shared memory (i.e. update state dict to parent process)
        for k in state_dict.keys():
            shared_dk = f'{k}_{worker_id}'
            shared_data = u.sm_array(shared_mems[shared_dk], data_shapes[shared_dk], data_dtypes[shared_dk])
            shared_data[...] = state_dict[k]
        return

    cur_adj = np.where(dist_wm[worker_id])[0]

    cur_logger = logging.getLogger()
    cur_logger.setLevel(logging.DEBUG)
    cur_logger.addHandler(QueueHandler(log_que))
    network = NeuralNetwork(device, model, task, worker_id, ftime, cur_logger)
    data_names, data_shapes, data_dtypes = shm_info
    network.load_state_dict(init_params)
    network.SetTrainSet()
    all_sms = start_sm()

    p2c_link.get()
    cnt = 0
    while cnt < round_num:
        # print(f'Worker {worker_id} start training...')
        network.Train(epoch=local_epoch_num, round_id=cnt)
        network.to('cpu')
        c2p_link.put((network.g_acc, network.sigma_c_new))
        upload_data_to_psm(network.state_dict(), all_sms)
        cur_logger.log(logging.INFO, f'Worker {worker_id} done.')
        agg_sigma_c = p2c_link.get()
        if ssl_method == 'DFLSemi':
            network.sigma_c_old = agg_sigma_c
        p2c_link.get()  # next round and get aggregated model
        # cur_adj in l_client_ids
        rand_adj_wid = np.random.choice(cur_adj)
        aggregated_params, adj_params = load_data_from_psm(network.state_dict(), all_sms, rand_adj_wid)
        network.cur_params, network.adj_params = aggregated_params, adj_params
        network.to(device)
        cnt += 1
    close_sm(all_sms)
    return


def Aggregation():
    def _update_sigma(cur_sigma_c, adj_sigma_c):
        for k in cur_sigma_c.keys():
            cur_sigma_c[k] = max(cur_sigma_c[k], adj_sigma_c[k])
        return cur_sigma_c

    # local_models = DistAvg(logger, current_local_models, dist_wm, round_idx)
    acc_list = g_acc_list if adap_agg else test_acc_list
    local_models = DistAvgAdaptive(logger, current_local_models, dist_wm, round_idx, acc_list)
    if ssl_method == 'DFLSemi':
        cur_sigma_list = [sigma_c_list[c][-1] for c in range(num_worker)]
        for ci in m_client_ids + u_client_ids:
            adj_ids = np.where(dist_wm[ci])[0]
            for cj in adj_ids:
                if cj in m_client_ids + u_client_ids:
                    cur_sigma_list[ci] = _update_sigma(cur_sigma_list[ci], cur_sigma_list[cj])
    else:
        cur_sigma_list = [None for _ in range(num_worker)]
    return local_models.copy(), cur_sigma_list


def ResWeightsConn():
    for ci in range(num_worker):
        cur_local_model = current_local_models[ci]
        last_local_model = record_ws[ci]
        res_w = {}
        for l in cur_local_model.keys():
            res_w[l] = 0.8 * cur_local_model[l] + 0.2 * last_local_model[l]
        current_local_models[ci] = res_w
    return current_local_models


# main process
if __name__ == '__main__':
    import warnings

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in gpu_list)
    warnings.filterwarnings("ignore", category=UserWarning)
    test_device = 'cuda:0'

    processing.set_start_method('spawn')
    config = init_global_config()
    log_dir = log_dir + f'/{task}'
    tmp_path = Path(log_dir)
    tmp_path.mkdir(exist_ok=True, parents=True)
    config.log_dir = log_dir

    # =================== global variables/containers ====================
    # logging queue
    log_queue = processing.Queue(-1)
    logger_process = processing.Process(target=Logging, args=(log_queue, u.get_logger, config))
    logger_process.start()
    file_time = log_queue.get()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(QueueHandler(log_queue))
    # pool to store parallel threading of training or attacking
    process_pool = []

    # send data from parent to child or child to parent
    p2c_links, c2p_links = [], []

    # local training loss
    # =================== INITIALIZATION ====================
    logger.log(INFO, f"__file__:    {__file__}")
    logger.log(INFO, 'Config parameters: ')
    logger.log(INFO, config)
    # checking global parameters
    logger.log(INFO, 'Checking global parameters......')
    assert num_worker == len(dist_wm)

    logger.log(INFO, 'Initializing global model container......')
    test_model = NeuralNetwork(device=test_device, model=model, task=task, logger=logger)
    state_keys = test_model.state_dict().keys()
    test_model.SetTestSet()

    # create shared memory
    all_shr_mems, buf_infos = create_shared_memory(test_model)

    test_model.to('cpu')
    prev_local_models = [test_model.state_dict().copy() for _ in range(num_worker)]

    # creating workers
    print('Creating workers')
    for i in range(0, num_worker):
        cur_device = f'cuda:{i % len(gpu_list)}'
        try:
            logger.log(INFO, f'Communication link of worker {i}......')
            p2c_links.append(processing.SimpleQueue())
            c2p_links.append(processing.SimpleQueue())
            # creating a benign worker process
            process_pool.append(processing.Process(target=LocalTraining, args=(
                i, test_model.state_dict().copy(), buf_infos, file_time, p2c_links[i], c2p_links[i],
                log_queue, cur_device)))
            time.sleep(0.1)
        except:
            print('\033[31mFailed\033[0m')
            sys.exit(-1)
    # activate worker processes
    for i in range(0, num_worker):
        try:
            logger.log(INFO, f'Activating worker {i}......')
            process_pool[i].start()
        except:
            print('\033[31mFailed\033[0m')
            sys.exit(-1)
    # =================== INITIALIZATION ====================

    # =================== Server process ====================
    test_acc_list = [[] for _ in range(num_worker)]
    g_acc_list = [[] for _ in range(num_worker)]
    sigma_c_list = [[] for _ in range(num_worker)]
    avg_test_accs = []
    for link in p2c_links:
        link.put('start')
    logger.log(INFO, '\nTraining Start!')
    iter_time = u.AverageMeter()

    for round_idx in range(0, round_num):
        end = time.time()
        logger.log(INFO, f'Global round {round_idx + 1}......')
        start_time = time.perf_counter()
        current_local_models = []
        for pi, c2p in enumerate(c2p_links):
            g_acc, sigma_c = c2p.get()
            sigma_c_list[pi].append(sigma_c)
            if g_acc is not None:
                g_acc_list[pi].append(g_acc)
            model_param = load_data_from_csm(all_shr_mems, pi, buf_infos[1], buf_infos[2], state_keys)
            current_local_models.append(model_param)

        current_local_models, agg_sigma_list = Aggregation()
        if ssl_method == 'CBAFed':
            if round_idx == 0:
                record_ws = copy.deepcopy(current_local_models)
            if round_idx % 5 == 0:
                current_local_models = ResWeightsConn()
                record_ws = copy.deepcopy(current_local_models)
        if (round_idx + 1) % test_round_interval == 0:
            test_model.to(test_device)
            for i, local_model in enumerate(current_local_models):
                test_model.load_state_dict(local_model.copy())
                logger.log(INFO, f"worker {i}")
                if (round_idx + 1) >= (round_num - test_round_interval * 10):
                    test_acc = test_model.Test()
                else:
                    test_acc = test_model.TestOneBatch(round_idx, i)
                test_acc_list[i].append(test_acc)
            test_model.to('cpu')
            logger.log(INFO, f"The average accuracy of benign clients on benign set: ")
            mean_test_acc = np.mean([test_acc_list[i][-1] for i in range(num_worker)])
            avg_test_accs.append(mean_test_acc)
            logger.log(INFO, f"{mean_test_acc:.4f}")

        for i, p2c in enumerate(p2c_links):
            p2c.put(agg_sigma_list[i])
            p2c.put('next round')
            upload_data_to_csm(all_shr_mems, i, buf_infos[1], buf_infos[2], current_local_models[i].copy())
            prev_local_models[i] = current_local_models[i].copy()
        logger.log(INFO, dist_wm)
        iter_time.update(time.time() - end)
        logger.log(INFO, f'Round: [{round_idx + 1}/{round_num}], '
                         f'Round time {iter_time.val:.3f} ({iter_time.avg:.3f}) '
                         f'Remain {u.calc_remain_time(round_num, round_idx, iter_time)} '
                   )
    # =================== Server process ====================
    logger.log(INFO, "=================== Statistics ====================")
    final_accs = []
    for wi, b_acc in enumerate(test_acc_list):
        logger.log(INFO, f"Worker: {wi}")
        logger.log(INFO, b_acc)
        final_accs.append(b_acc[-10:])
    final_accs = np.array(final_accs)
    logger.log(INFO, f"The average accuracy of all clients for each round: ")
    logger.log(INFO, avg_test_accs)
    logger.log(INFO, f"The average accuracy of all agents in last 10 test rounds: ")
    logger.log(INFO, f"{np.mean(avg_test_accs[-10:]) * 100:.2f}%")
    logger.log(INFO, f"{np.mean(np.mean(final_accs, axis=1), axis=0) * 100:.2f}%")
    logger.log(INFO, f"The std accuracy of all agents in last 10 test rounds: ")
    logger.log(INFO, f"{np.std(np.mean(final_accs, axis=1), axis=0) * 100:.2f}%")
    print('Terminating all processes...')
    # terminate logging process
    log_queue.put_nowait(None)
    for p in process_pool:
        p.terminate()
    close_shared_memory(all_shr_mems)
