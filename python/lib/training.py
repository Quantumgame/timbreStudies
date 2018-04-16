# Copyright (c) Baptiste Caramiaux, Etienne Thoret
# All rights reserved

import numpy as np
import matplotlib.pylab as plt
import pickle
import os


def kernel_optim(input_data,
                 target_data,
                 cost='correlation',
                 init_sig_mean=10.0,
                 init_sig_var=0.5,
                 num_loops=10000,
                 learning_rate=0.001,
                 log_foldername='./',
                 logging=False,
                 verbose=True):

    if (verbose):
        print("* training sigmas of gaussian kernels with cost '{}'".format(
            cost))
    # plt.plot(input_data)
    # plt.show()
    ndims, ninstrus = input_data.shape[0], input_data.shape[1]
    no_samples = ninstrus * (ninstrus - 1) / 2
    sigmas = np.abs(init_sig_mean + init_sig_var * np.random.randn(ndims, 1))
    init_seed = sigmas
    gradients = np.zeros((ndims, 1))

    correlations = []  # = np.zeros((num_loops, ))

    idx_triu = np.triu_indices(target_data.shape[0], k=1)
    # print(idx_triu[0], idx_triu[1])
    target_v = target_data[idx_triu]
    mean_target = np.mean(target_v)
    std_target = np.std(target_v)

    kernel = np.zeros((ninstrus, ninstrus))
    dkernel = np.zeros((ninstrus, ninstrus, ndims))

    if logging:
        pickle.dump({
            'seed': init_seed,
            'args': {
                'cost': cost,
                'init_sig_mean': init_sig_mean,
                'init_sig_var': init_sig_var,
                'num_loops': num_loops,
                'log_foldername': log_foldername,
                'learning_rate': learning_rate
            }
        }, open(os.path.join(log_foldername, 'optim_config.pkl'), 'wb'))

    for loop in range(num_loops):  # 0 to nump_loops-1
        sigmas = sigmas - learning_rate * gradients * sigmas
        for i in range(ninstrus):
            # plt.plot(input_data[:, i])
            # plt.plot(np.divide(input_data[:, i], (sigmas[:, 0] + np.finfo(float).eps)))
            # plt.show()
            for j in range(i + 1, ninstrus):
                kernel[i, j] = np.exp(-np.sum(
                    np.power(
                        np.divide(input_data[:, i] - input_data[:, j],
                                  (sigmas[:, 0] + np.finfo(float).eps)), 2)))
                # print(kernel[i, j], np.sum(np.power(input_data[:, i] - input_data[:, j],2)), np.sum(
                #     np.power(
                #         np.divide(input_data[:, i] - input_data[:, j],
                #                   (sigmas[:, 0] + np.finfo(float).eps)), 2)), np.max(input_data[:, i]), np.max(input_data[:, j]))
                dkernel[i, j, :] = 2 * kernel[i, j] * np.power(
                    (input_data[:, i] - input_data[:, j]), 2) / (np.power(
                        sigmas[:, 0], 3) + np.finfo(float).eps)
        kernel_v = kernel[idx_triu]
        mean_kernel = np.mean(kernel_v)
        std_kernel = np.std(kernel_v)

        Jn = np.sum(
            np.multiply(kernel_v - mean_kernel, target_v - mean_target))
        Jd = (no_samples - 1) * std_target * std_kernel
        
        correlations.append(Jn / (Jd + np.finfo(float).eps))

        for k in range(ndims):
            # dkernel_k_v = dkernel[:, :, k][idx_triu]
            dkernel_k_v = dkernel[idx_triu[0], idx_triu[1], k]
            dJn = np.sum(dkernel_k_v * (target_v - mean_target))
            dJd = (no_samples - 1) / no_samples * \
                        std_target / (std_kernel + np.finfo(float).eps) * \
                        np.sum(dkernel_k_v * (kernel_v - mean_kernel))
            gradients[k] = (Jd * dJn - Jn * dJd) / (np.power(Jd,2) + np.finfo(float).eps)

        # dkernel_k_v = dkernel[idx_triu[0], idx_triu[1], :].reshape(-1, dkernel.shape[2])
        # # print(dkernel_k_v.shape, target_v.shape)
        # dJn = np.sum(np.multiply(dkernel_k_v , (target_v.reshape(-1,1) - mean_target)), axis=0)
        # dJd = (no_samples - 1) / no_samples * \
        #                 std_target / (std_kernel + np.finfo(float).eps) * \
        #                 np.sum(np.multiply(dkernel_k_v, (kernel_v.reshape(-1,1) - mean_kernel)), axis=0)
        # gradients = (Jd * dJn - Jn * dJd) / (Jd**2 + np.finfo(float).eps)

        monitoring_step = 5000
        if (verbose):
            if ((loop + 1) % monitoring_step == 0):
                print('  |_ loop num.: {}/{} | grad={:.6f} | J={:.6f}'.format(
                    loop + 1, num_loops,
                    np.linalg.norm(gradients, 2), correlations[loop]))

        if logging:
            if ((loop + 1) % monitoring_step == 0):
                pickle.dump({
                    'sigmas': sigmas,
                    'kernel': kernel,
                    'Jn': Jn,
                    'Jd': Jd,
                    'correlations': correlations,
                    'gradients': gradients
                }, open(os.path.join(log_foldername,
                    'optim_process_l={}.pkl'.format(loop+1)), 'wb'))

    return correlations



def kernel_optim_log(input_data,
                     target_data,
                     cost='correlation',
                     init_sig_mean=10.0,
                     init_sig_var=0.5,
                     num_loops=10000,
                     log_foldername='./',
                     logging=False,
                     verbose=True):

    if (verbose):
        print("* training sigmas of gaussian kernels with cost '{}'".format(
            cost))
    # plt.plot(input_data)
    # plt.show()
    ndims, ninstrus = input_data.shape[0], input_data.shape[1]
    no_samples = ninstrus * (ninstrus - 1) / 2
    sigmas = np.abs(init_sig_mean + init_sig_var * np.random.randn(ndims, 1))
    init_seed = sigmas
    gradients = np.zeros((ndims, 1))

    correlations = []  # = np.zeros((num_loops, ))

    idx_triu = np.triu_indices(target_data.shape[0], k=1)
    # print(idx_triu[0], idx_triu[1])
    target_v = target_data[idx_triu]
    mean_target = np.mean(target_v)
    std_target = np.std(target_v)

    logkernel = np.zeros((ninstrus, ninstrus))
    dlogkernel = np.zeros((ninstrus, ninstrus, ndims))

    learning_rate = 0.1

    if logging:
        pickle.dump({
            'seed': init_seed,
            'args': {
                'cost': cost,
                'init_sig_mean': init_sig_mean,
                'init_sig_var': init_sig_var,
                'num_loops': num_loops,
                'log_foldername': log_foldername
            }
        }, open(os.path.join(log_foldername, 'optim_config.pkl'), 'wb'))

    for loop in range(num_loops):  # 0 to nump_loops-1
        sigmas = sigmas - learning_rate * gradients * sigmas
        for i in range(ninstrus):
            for j in range(i + 1, ninstrus):
                logkernel[i, j] = -np.sum(np.power(np.divide(input_data[:, i] - input_data[:, j], (sigmas[:, 0] + np.finfo(float).eps)), 2))
                dlogkernel[i, j, :] = 2 * np.power((input_data[:, i] - input_data[:, j]), 2) / (np.power(sigmas[:, 0], 3) + np.finfo(float).eps)
        logkernel_v = logkernel[idx_triu]
        mean_logkernel = np.mean(logkernel_v)
        std_logkernel = np.std(logkernel_v)

        # print(logkernel_v.shape, mean_logkernel)
        # print(logkernel_v)
        Jn = np.sum(np.multiply(logkernel_v - mean_logkernel, target_v - mean_target))
        Jd = (no_samples - 1) * std_target * std_logkernel
        # Jd = std_target * std_logkernel
        
        # print(Jn, Jd, std_target, std_logkernel)

        correlations.append(Jn / (Jd + np.finfo(float).eps))

        for k in range(ndims):
            dlogkernel_k_v = dlogkernel[idx_triu[0], idx_triu[1], k]
            dJn = np.sum(dlogkernel_k_v * (target_v - mean_target))
            dJd = (no_samples - 1) / no_samples * std_target / (std_logkernel + np.finfo(float).eps) * np.sum(dlogkernel_k_v * (logkernel_v - mean_logkernel))
            # dJd = std_target / (std_logkernel + np.finfo(float).eps) * np.sum(dlogkernel_k_v * (logkernel_v - mean_logkernel))
            gradients[k] = (Jd * dJn - Jn * dJd) / (np.power(Jd,2) + np.finfo(float).eps)
        # verbose
        if (verbose):
            if ((loop + 1) % 1000 == 0):
                print('  |_ loop num.: {}/{} | grad={:.6f} | J={:.6f}'.format(
                    loop + 1, num_loops,
                    np.linalg.norm(gradients, 2), correlations[loop]))

        if logging:
            if ((loop + 1) % 1000 == 0):
                pickle.dump({
                    'sigmas': sigmas,
                    'logkernel': logkernel,
                    'Jn': Jn,
                    'Jd': Jd,
                    'correlations': correlations,
                    'gradients': gradients
                }, open(os.path.join(log_foldername,
                    'optim_process_l={}.pkl'.format(loop+1)), 'wb'))

    return correlations


def resume_kernel_optim(retrieve_foldername='./',
                        log_foldername='./',
                        num_loops = 0,
                        logging = False,
                        verbose = True):
    # load file
    for root, dirs, files in os.walk(retrieve_foldername):
        loop_id = []
        for name in files:
            if name.split('.')[-1] == 'pkl' and 'optim_process' in name.split('.')[0]:
                loop_id.append(int(name.split('.')[0].split('=')[-1]))
    retrieved_loop = np.max(loop_id)
    filename = os.path.join(retrieve_foldername,'optim_process_l={}.pkl'.format(retrieved_loop))
    optim_process = pickle.load(open(filename, 'rb'))
    optim_config = pickle.load(open(os.path.join(retrieve_foldername,'optim_config.pkl'), 'rb'))
    dataset = pickle.load(open(os.path.join(retrieve_foldername,'dataset.pkl'), 'rb'))

    input_data = dataset['data_proj']
    target_data = dataset['dissimilarities']

    if verbose:
        print("* resuming with '{}' of size {}".format(retrieve_foldername.split('/')[-1], input_data.shape))

    init_seed = optim_config['seed']
    cost = optim_config['args']['cost']
    init_sig_mean = optim_config['args']['init_sig_mean']
    init_sig_var = optim_config['args']['init_sig_var']

    if (verbose):
        if num_loops == 0:
            print("* retrieving training sigmas of gaussian kernels with cost '{}' at loop n.{}/{}".format(
                cost, retrieved_loop, optim_config['args']['num_loops']))
        else:
            if retrieved_loop < num_loops:
                print("* retrieving training sigmas of gaussian kernels with cost '{}' at loop n.{}, extending with num loops={}".format(
                    cost, retrieved_loop, num_loops))
            else:
                print("* retrieving training sigmas of gaussian kernels with cost '{}' at loop n.{}/{}, nothing to do.".format(
                    cost, retrieved_loop, num_loops))

    if (num_loops == 0):
        num_loops = optim_config['args']['num_loops'] #- retrieved_loop

    ndims, ninstrus = input_data.shape[0], input_data.shape[1]
    no_samples = ninstrus * (ninstrus - 1) / 2
    sigmas = optim_process['sigmas']
    gradients = optim_process['gradients']
    correlations = optim_process['correlations']

    idx_triu = np.triu_indices(target_data.shape[0], k=1)
    target_v = target_data[idx_triu]
    mean_target = np.mean(target_v)
    std_target = np.std(target_v)

    kernel = np.zeros((ninstrus, ninstrus))
    dkernel = np.zeros((ninstrus, ninstrus, ndims))

    learned_sigmas = []
    for loop in range(retrieved_loop, num_loops):  # 0 to nump_loops-1
        sigmas = sigmas - gradients * sigmas
        for i in range(ninstrus):
            for j in range(i + 1, ninstrus):
                kernel[i, j] = np.exp(-np.sum(
                    np.power(
                        np.divide(input_data[:, i] - input_data[:, j],
                                  sigmas[:, 0]), 2)))
                dkernel[i, j, :] = 2 * kernel[i, j] * np.power(
                    (input_data[:, i] - input_data[:, j]), 2) / np.power(
                        sigmas[:, 0], 3)
        kernel_v = kernel[idx_triu]
        mean_kernel = np.mean(kernel_v)
        std_kernel = np.std(kernel_v)

        Jn = np.sum(
            np.multiply(kernel_v - mean_kernel, target_v - mean_target))
        Jd = (no_samples - 1) * std_target * std_kernel
        correlations.append(Jn / Jd)

        for k in range(ndims):
            dkernel_k_v = dkernel[:, :, k][idx_triu]
            dJn = np.sum(dkernel_k_v * (target_v - mean_target))
            dJd = (no_samples - 1) / no_samples * \
                        std_target / std_kernel * \
                        np.sum(dkernel_k_v * (kernel_v - mean_kernel))
            gradients[k] = (Jd * dJn - Jn * dJd) / (Jd**2)
        # verbose
        if (verbose):
            if ((loop + 1) % 1000 == 0):
                print('  |_ loop num.: {}/{} | grad={:.6f} | J={:.6f}'.format(
                    loop + 1, num_loops,
                    np.linalg.norm(gradients, 2), correlations[loop]))
                learned_sigmas.append(sigmas)

        if logging:
            if ((loop + 1) % 1000 == 0):
                pickle.dump({
                    'sigmas': sigmas,
                    'kernel': kernel,
                    'Jn': Jn,
                    'Jd': Jd,
                    'correlations': correlations,
                    'gradients': gradients
                }, open(os.path.join(log_foldername,
                    'optim_process_l={}.pkl'.format(loop+1)), 'wb'))

    return correlations, learned_sigmas
