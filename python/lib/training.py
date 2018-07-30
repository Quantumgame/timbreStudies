# Copyright (c) Baptiste Caramiaux, Etienne Thoret
# All rights reserved

import numpy as np
import matplotlib.pylab as plt
import pickle
import os
from scipy.optimize import minimize


def kernel_optim(input_data,
                 target_data,
                 cost='correlation',
                 loss='exp_sum',
                 init_sig_mean=10.0,
                 init_sig_var=0.5,
                 num_loops=10000,
                 learning_rate=0.001,
                 log_foldername='./',
                 resume=None,
                 logging=False,
                 verbose=True):

    if (verbose):
        print("* training sigmas of gaussian kernels with cost '{}'".format(
            cost))
    # plt.plot(input_data)
    # plt.show()
    ndims, ninstrus = input_data.shape[0], input_data.shape[1]
    no_samples = ninstrus * (ninstrus - 1) / 2
    if resume != None:
        init_seed = resume['init_seed']
        sigmas = resume['sigmas']
        gradients = resume['gradients']
        correlations = resume['correlations']
        retrieved_loop = resume['retrieved_loop']
    else:
        sigmas = np.abs(init_sig_mean + init_sig_var * np.random.randn(ndims, 1))
        init_seed = sigmas
        gradients = np.zeros((ndims, 1))
        correlations = []  # = np.zeros((num_loops, ))
        retrieved_loop = 0

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
                'loss': loss,
                'init_sig_mean': init_sig_mean,
                'init_sig_var': init_sig_var,
                'num_loops': num_loops,
                'log_foldername': log_foldername,
                'learning_rate': learning_rate
            }
        }, open(os.path.join(log_foldername, 'optim_config.pkl'), 'wb'))


    # print(input_data[:,3].shape)
    # plt.plot(input_data[:,3])
    # plt.plot(input_data[:,3].reshape(-1,1) / sigmas)
    # plt.show()

    for loop in range(retrieved_loop, num_loops):  # 0 to nump_loops-1
        sigmas = sigmas - learning_rate * gradients * sigmas
        for i in range(ninstrus):
            # plt.plot(input_data[:, i])
            # plt.plot(np.divide(input_data[:, i], (sigmas[:, 0] + np.finfo(float).eps)))
            # plt.show()
            for j in range(i + 1, ninstrus):
                if loss == 'exp_sum':
                    kernel[i, j] = np.exp(-np.sum(
                        np.power(
                            np.divide(input_data[:, i] - input_data[:, j],
                                      (sigmas[:, 0] + np.finfo(float).eps)), 2)))
                    dkernel[i, j, :] = 2 * kernel[i, j] * np.power(
                        (input_data[:, i] - input_data[:, j]), 2) / (np.power(
                            sigmas[:, 0], 3) + np.finfo(float).eps)
                elif loss == 'sum':
                    kernel[i, j] = -np.sum(
                        np.power(
                            np.divide(input_data[:, i] - input_data[:, j],
                                      (sigmas[:, 0] + np.finfo(float).eps)), 2))
                    dkernel[i, j, :] = 2 * np.power((input_data[:, i] - input_data[:, j]), 2) / (np.power(
                            sigmas[:, 0], 3) + np.finfo(float).eps)
                # elif loss == 'log_likelihood':
                #     kernel[i, j] = -np.sum(
                #         np.power(
                #             np.divide(input_data[:, i] - input_data[:, j],
                #                       (sigmas[:, 0] + np.finfo(float).eps)), 2))
                #     dkernel[i, j, :] = 2 * np.power((input_data[:, i] - input_data[:, j]), 2) / (np.power(
                #             sigmas[:, 0], 3) + np.finfo(float).eps)

        # plt.subplot(1,2,1)
        # plt.imshow(kernel)
        # plt.subplot(1,2,2)
        # plt.imshow(target_data)
        # plt.show()

        kernel_v = kernel[idx_triu]
        mean_kernel = np.mean(kernel_v)
        std_kernel = np.std(kernel_v)

        Jn = np.sum(np.multiply(kernel_v - mean_kernel, target_v - mean_target))
        Jd = (no_samples - 1) * std_target * std_kernel
        
        correlations.append(Jn / (Jd + np.finfo(float).eps))

        for k in range(ndims):
            dkernel_k_v = dkernel[:, :, k][idx_triu]
            # dkernel_k_v = dkernel[idx_triu[0], idx_triu[1], k]
            dJn = np.sum(dkernel_k_v * (target_v - mean_target))
            dJd = (no_samples - 1) / no_samples * \
                        std_target / (std_kernel + np.finfo(float).eps) * \
                        np.sum(dkernel_k_v * (kernel_v - mean_kernel))
            gradients[k] = (Jd * dJn - Jn * dJd) / (np.power(Jd,2) + np.finfo(float).eps)
            # print(loop+1, k, Jd, dJn, Jn, dJd, (np.power(Jd,2) + np.finfo(float).eps), np.linalg.norm(gradients, 2))

        # dkernel_k_v = dkernel[idx_triu[0], idx_triu[1], :].reshape(-1, dkernel.shape[2])
        # # print(dkernel_k_v.shape, target_v.shape)
        # dJn = np.sum(np.multiply(dkernel_k_v , (target_v.reshape(-1,1) - mean_target)), axis=0)
        # dJd = (no_samples - 1) / no_samples * \
        #                 std_target / (std_kernel + np.finfo(float).eps) * \
        #                 np.sum(np.multiply(dkernel_k_v, (kernel_v.reshape(-1,1) - mean_kernel)), axis=0)
        # gradients = (Jd * dJn - Jn * dJd) / (Jd**2 + np.finfo(float).eps)

        # print('  |_ loop num.: {}/{} | grad={:.6f} | J={:.6f}'.format(loop + 1, num_loops, np.linalg.norm(gradients, 2), correlations[loop]))
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





def kernel_optim_lbfgs(input_data,
                       target_data,
                       cost='correlation',
                       loss='exp_sum',
                       init_sig_mean=10.0,
                       init_sig_var=0.5,
                       num_loops=10000,
                       method='L-BFGS-B',
                       log_foldername='./',
                       resume=None,
                       logging=False,
                       verbose=True):

    if (verbose):
        print("* training sigmas of gaussian kernels with cost '{}' and method '{}'".format(
            cost, method))
    
    ndims, ninstrus = input_data.shape[0], input_data.shape[1]
    no_samples = ninstrus * (ninstrus - 1) / 2
    if resume != None:
        init_seed = resume['init_seed']
        sigmas = resume['sigmas']
        gradients = resume['gradients']
        correlations = resume['correlations']
        retrieved_loop = resume['retrieved_loop']
    else:
        sigmas = np.abs(init_sig_mean + init_sig_var * np.random.randn(ndims, 1))
        init_seed = sigmas
        gradients = np.zeros((ndims, 1))
        correlations = []  # = np.zeros((num_loops, ))
        retrieved_loop = 0

    ndims, ninstrus = input_data.shape[0], input_data.shape[1]
    no_samples = ninstrus * (ninstrus - 1) / 2
    idx_triu = np.triu_indices(target_data.shape[0], k=1)
    target_v = target_data[idx_triu]
    mean_target = np.mean(target_v)
    std_target = np.std(target_v)

    loop_cpt = 0
    correlations = []

    optim_options = {'disp': None, 'maxls': 50, 'iprint': -1, 'gtol': 1e-36, 'eps': 1e-16, 'maxiter': num_loops, 'ftol': 1e-36}
    optim_bounds = [(1.0*f,1e5*f) for f in np.ones((ndims,))]

    if logging:
        pickle.dump({
            'seed': init_seed,
            'cost': cost,
            'loss': loss,
            'method': method,
            'init_sig_mean': init_sig_mean,
            'init_sig_var': init_sig_var,
            'num_loops': num_loops,
            'log_foldername': log_foldername,
            'optim_options': {'options': optim_options, 'bounds': optim_bounds}
        }, open(os.path.join(log_foldername, 'optim_config.pkl'), 'wb'))

    def corr(x):
        kernel = np.zeros((ninstrus, ninstrus))
        for i in range(ninstrus):
                for j in range(i + 1, ninstrus):
                    kernel[i, j] = np.exp(-np.sum(
                        np.power(
                            np.divide(input_data[:, i] - input_data[:, j],
                                      (x + np.finfo(float).eps)), 2)))
        kernel_v = kernel[idx_triu]
        mean_kernel = np.mean(kernel_v)
        std_kernel = np.std(kernel_v)
        Jn = np.sum(np.multiply(kernel_v - mean_kernel, target_v - mean_target))
        Jd = (no_samples - 1) * std_target * std_kernel
        return Jn/Jd


    def grad_corr(x):
        kernel = np.zeros((ninstrus, ninstrus))
        dkernel = np.zeros((ninstrus, ninstrus, ndims))
        for i in range(ninstrus):
                for j in range(i + 1, ninstrus):
                    kernel[i, j] = np.exp(-np.sum(
                        np.power(
                            np.divide(input_data[:, i] - input_data[:, j],
                                      (x + np.finfo(float).eps)), 2)))
                    dkernel[i, j, :] = 2 * kernel[i, j] * np.power(
                        (input_data[:, i] - input_data[:, j]), 2) / (np.power(x, 3) + np.finfo(float).eps)
        kernel_v = kernel[idx_triu]
        mean_kernel = np.mean(kernel_v)
        std_kernel = np.std(kernel_v)
        Jn = np.sum(np.multiply(kernel_v - mean_kernel, target_v - mean_target))
        Jd = (no_samples - 1) * std_target * std_kernel
        for k in range(ndims):
            dkernel_k_v = dkernel[:, :, k][idx_triu]
            dJn = np.sum(dkernel_k_v * (target_v - mean_target))
            dJd = (no_samples - 1) / no_samples * \
                        std_target / (std_kernel + np.finfo(float).eps) * \
                        np.sum(dkernel_k_v * (kernel_v - mean_kernel))
            gradients[k] = (Jd * dJn - Jn * dJd) / (np.power(Jd,2) + np.finfo(float).eps)
        return gradients


    def print_corr(xk):
        kernel = np.zeros((ninstrus, ninstrus))
        for i in range(ninstrus):
                for j in range(i + 1, ninstrus):
                    kernel[i, j] = np.exp(-np.sum(
                        np.power(
                            np.divide(input_data[:, i] - input_data[:, j],
                                      (xk + np.finfo(float).eps)), 2)))
        kernel_v = kernel[idx_triu]
        mean_kernel = np.mean(kernel_v)
        std_kernel = np.std(kernel_v)
        Jn = np.sum(np.multiply(kernel_v - mean_kernel, target_v - mean_target))
        Jd = (no_samples - 1) * std_target * std_kernel
        
        if not os.path.isfile(os.path.join(log_foldername, 'tmp.pkl')):
            loop_cpt = 1
            pickle.dump({'loop': loop_cpt, 'correlation': [Jn/Jd]}, open(os.path.join(log_foldername, 'tmp.pkl'), 'wb'))
            correlations = [Jn/Jd]
            pickle.dump({
                    'sigmas': xk,
                    'kernel': kernel,
                    'Jn': Jn,
                    'Jd': Jd,
                    'correlations': correlations,
                }, open(os.path.join(log_foldername,
                    'optim_process_l={}.pkl'.format(loop_cpt)), 'wb'))
        else:
            last_loop = pickle.load(open(os.path.join(log_foldername,'tmp.pkl'), 'rb'))
            loop_cpt = last_loop['loop'] + 1
            correlations = last_loop['correlation']
            correlations.append(Jn/Jd)
            monitoring_step = 25
            if (loop_cpt % monitoring_step == 0):
                print('  |_ loop={} J={:.6f}'.format(loop_cpt, Jn/Jd))
                pickle.dump({
                    'sigmas': xk,
                    'kernel': kernel,
                    'Jn': Jn,
                    'Jd': Jd,
                    'correlations': correlations,
                }, open(os.path.join(log_foldername,
                    'optim_process_l={}.pkl'.format(loop_cpt)), 'wb'))
            pickle.dump({'loop': loop_cpt, 'correlation': correlations}, open(os.path.join(log_foldername, 'tmp.pkl'), 'wb'))

    res = minimize(corr, sigmas, args=(), method=method, jac=grad_corr, callback=print_corr, options=optim_options, bounds=optim_bounds)

    return correlations


def kernel_optim_lbfgs_log(input_data,
                       target_data,
                       cost='correlation',
                       loss='exp_sum',
                       init_sig_mean=10.0,
                       init_sig_var=0.5,
                       num_loops=10000,
                       method='L-BFGS-B',
                       log_foldername='./',
                       resume=None,
                       logging=False,
                       verbose=True):
    
    if (verbose):
        print("* training sigmas of gaussian kernels with cost '{}' and method '{}'".format(
            cost, method))
    
    ndims, ninstrus = input_data.shape[0], input_data.shape[1]
    no_samples = ninstrus * (ninstrus - 1) / 2
    if resume != None:
        init_seed = resume['init_seed']
        sigmas = resume['sigmas']
        gradients = resume['gradients']
        correlations = resume['correlations']
        retrieved_loop = resume['retrieved_loop']
    else:
        sigmas = np.abs(init_sig_mean + init_sig_var * np.random.randn(ndims, 1))
        init_seed = sigmas
        gradients = np.zeros((ndims, 1))
        correlations = []  # = np.zeros((num_loops, ))
        retrieved_loop = 0

    ndims, ninstrus = input_data.shape[0], input_data.shape[1]
    no_samples = ninstrus * (ninstrus - 1) / 2
    idx_triu = np.triu_indices(target_data.shape[0], k=1)
    target_v = target_data[idx_triu]
    mean_target = np.mean(target_v)
    std_target = np.std(target_v)

    loop_cpt = 0
    correlations = []

    optim_options = {'disp': None, 'maxls': 50, 'iprint': -1, 'gtol': 1e-36, 'eps': 1e-8, 'maxiter': num_loops, 'ftol': 1e-36}
    optim_bounds = [(1.0*f,1e5*f) for f in np.ones((ndims,))]

    if logging:
        pickle.dump({
            'seed': init_seed,
            'cost': cost,
            'loss': loss,
            'method': method,
            'init_sig_mean': init_sig_mean,
            'init_sig_var': init_sig_var,
            'num_loops': num_loops,
            'log_foldername': log_foldername,
            'optim_options': {'options': optim_options, 'bounds': optim_bounds}
        }, open(os.path.join(log_foldername, 'optim_config.pkl'), 'wb'))

    def corr(x):
        kernel = np.zeros((ninstrus, ninstrus))
        for i in range(ninstrus):
                for j in range(i + 1, ninstrus):
                    kernel[i, j] = -np.sum(
                        np.power(
                            np.divide(input_data[:, i] - input_data[:, j],
                                      (x + np.finfo(float).eps)), 2))
        kernel_v = kernel[idx_triu]
        mean_kernel = np.mean(kernel_v)
        std_kernel = np.std(kernel_v)
        Jn = np.sum(np.multiply(kernel_v - mean_kernel, target_v - mean_target))
        # Jd = (no_samples - 1) * std_target * std_kernel
        Jd = no_samples * std_target * std_kernel
        return Jn/Jd


    def grad_corr(x):
        kernel = np.zeros((ninstrus, ninstrus))
        dkernel = np.zeros((ninstrus, ninstrus, ndims))
        for i in range(ninstrus):
                for j in range(i + 1, ninstrus):
                    kernel[i, j] = -np.sum(
                        np.power(
                            np.divide(input_data[:, i] - input_data[:, j],
                                      (x + np.finfo(float).eps)), 2))
                    dkernel[i, j, :] = 2 * np.power((input_data[:, i] - input_data[:, j]), 2) / (np.power(x, 3) + np.finfo(float).eps)
        kernel_v = kernel[idx_triu]
        mean_kernel = np.mean(kernel_v)
        std_kernel = np.std(kernel_v)
        Jn = np.sum(np.multiply(kernel_v - mean_kernel, target_v - mean_target))
        # Jd = (no_samples - 1) * std_target * std_kernel
        Jd = no_samples * std_target * std_kernel
        for k in range(ndims):
            dkernel_k_v = dkernel[:, :, k][idx_triu]
            dJn = np.sum(dkernel_k_v * (target_v - mean_target))
            # dJd = (no_samples - 1) / no_samples * \
            #             std_target / (std_kernel + np.finfo(float).eps) * \
            #             np.sum(dkernel_k_v * (kernel_v - mean_kernel))
            dJd = no_samples / no_samples * \
                        std_target / (std_kernel + np.finfo(float).eps) * \
                        np.sum(dkernel_k_v * (kernel_v - mean_kernel))
            gradients[k] = (Jd * dJn - Jn * dJd) / (np.power(Jd,2) + np.finfo(float).eps)
        return gradients


    def print_corr(xk):
        kernel = np.zeros((ninstrus, ninstrus))
        for i in range(ninstrus):
                for j in range(i + 1, ninstrus):
                    kernel[i, j] = np.exp(-np.sum(
                        np.power(
                            np.divide(input_data[:, i] - input_data[:, j],
                                      (xk + np.finfo(float).eps)), 2)))
        kernel_v = kernel[idx_triu]
        mean_kernel = np.mean(kernel_v)
        std_kernel = np.std(kernel_v)
        Jn = np.sum(np.multiply(kernel_v - mean_kernel, target_v - mean_target))
        # Jd = (no_samples - 1) * std_target * std_kernel
        Jd = no_samples * std_target * std_kernel
        
        if not os.path.isfile(os.path.join(log_foldername, 'tmp.pkl')):
            loop_cpt = 1
            pickle.dump({'loop': loop_cpt, 'correlation': [Jn/Jd]}, open(os.path.join(log_foldername, 'tmp.pkl'), 'wb'))
            correlations = [Jn/Jd]
            pickle.dump({
                    'sigmas': xk,
                    'kernel': kernel,
                    'Jn': Jn,
                    'Jd': Jd,
                    'correlations': correlations,
                }, open(os.path.join(log_foldername,
                    'optim_process_l={}.pkl'.format(loop_cpt)), 'wb'))
        else:
            last_loop = pickle.load(open(os.path.join(log_foldername,'tmp.pkl'), 'rb'))
            loop_cpt = last_loop['loop'] + 1
            correlations = last_loop['correlation']
            correlations.append(Jn/Jd)
            monitoring_step = 100
            if (loop_cpt % monitoring_step == 0):
                print('  |_ loop={} J={:.6f}'.format(loop_cpt, Jn/Jd))
                pickle.dump({
                    'sigmas': xk,
                    'kernel': kernel,
                    'Jn': Jn,
                    'Jd': Jd,
                    'correlations': correlations,
                }, open(os.path.join(log_foldername,
                    'optim_process_l={}.pkl'.format(loop_cpt)), 'wb'))
            pickle.dump({'loop': loop_cpt, 'correlation': correlations, 'sigmas': xk}, open(os.path.join(log_foldername, 'tmp.pkl'), 'wb'))

    res = minimize(corr, sigmas, args=(), method=method, jac=grad_corr, callback=print_corr, options=optim_options, bounds=optim_bounds)
    last_loop = pickle.load(open(os.path.join(log_foldername,'tmp.pkl'), 'rb'))
    sigmas_ = last_loop['sigmas']
    return correlations, sigmas_

def kernel_optim_lbfgs_log_crossts(input_data,
                                   target_data,
                                   cost='correlation',
                                   loss='exp_sum',
                                   init_sig_mean=10.0,
                                   init_sig_var=0.5,
                                   num_loops=10000,
                                   method='L-BFGS-B',
                                   log_foldername='./',
                                   logging=False,
                                   verbose=True):
    
    if (verbose):
        print("* training sigmas of gaussian kernels with cost '{}' and method '{}'".format(
            cost, method))

    tsps = list(sorted(input_data.keys()))
    ndims = input_data[tsps[0]].shape[0]  # dim of audio repres
    nsnds_tsp = [0]
    nsnds_tsp.extend([input_data[k].shape[1] for k in tsps])
    ninstrus = np.sum([input_data[k].shape[1] for k in tsps])   # total num of sounds
    # no_samples = ninstrus * (ninstrus - 1) / 2

    sigmas = np.abs(init_sig_mean + init_sig_var * np.random.randn(ndims, 1))
    init_seed = sigmas
    gradients = np.zeros((ndims, 1))
    correlations = []  # = np.zeros((num_loops, ))
    retrieved_loop = 0

    # target_data_all = np.zeros((ninstrus, ninstrus))
    # for ti, tsp in enumerate(tsps):
    #     offset = np.cumsum(nsnds_tsp[:ti+1])[-1]
    #     ninstrus_tsp = nsnds_tsp[ti+1]
    #     target_data_all[offset:offset+ninstrus_tsp,offset:offset+ninstrus_tsp] = target_data[tsp]
    # idx_triu = np.triu_indices(target_data_all.shape[0], k=1)
    # target_v = target_data_all[idx_triu]
    # mean_target = np.mean(target_v)
    # std_target = np.std(target_v)

    loop_cpt = 0
    correlations = []

    optim_options = {'disp': None, 'maxls': 50, 'iprint': -1, 'gtol': 1e-36, 'eps': 1e-36, 'maxiter': num_loops, 'ftol': 1e-36}
    optim_bounds = [(1.0*f,1e5*f) for f in np.ones((ndims,))]

    if logging:
        pickle.dump({
            'seed': init_seed,
            'cost': cost,
            'loss': loss,
            'method': method,
            'init_sig_mean': init_sig_mean,
            'init_sig_var': init_sig_var,
            'num_loops': num_loops,
            'log_foldername': log_foldername,
            'optim_options': {'options': optim_options, 'bounds': optim_bounds}
        }, open(os.path.join(log_foldername, 'optim_config.pkl'), 'wb'))

    def corr(x):
        corr_sum = 0
        for ti, tsp in enumerate(tsps):
            # offset = np.cumsum(nsnds_tsp[:ti+1])[-1]
            ninstrus_tsp = nsnds_tsp[ti+1]
            no_samples = ninstrus_tsp * (ninstrus_tsp - 1) / 2
            kernel = np.zeros((ninstrus_tsp, ninstrus_tsp))
            for i in range(ninstrus_tsp):
                for j in range(i + 1, ninstrus_tsp):
                    kernel[i, j] = -np.sum(
                        np.power(
                            np.divide(input_data[tsp][:, i] - input_data[tsp][:, j],
                                      (x + np.finfo(float).eps)), 2))
            idx_triu = np.triu_indices(target_data[tsp].shape[0], k=1)
            target_v = target_data[tsp][idx_triu]
            mean_target = np.mean(target_v)
            std_target = np.std(target_v)
            kernel_v = kernel[idx_triu]
            mean_kernel = np.mean(kernel_v)
            std_kernel = np.std(kernel_v)
            Jn = np.sum(np.multiply(kernel_v - mean_kernel, target_v - mean_target))
            Jd = no_samples * std_target * std_kernel
            corr_sum += Jn/Jd
        # print(corr_sum/len(tsps))
        return corr_sum #Jn/Jd

    def grad_corr(x):
        corr_sum = 0.0
        kernel_v = {}
        dkernel = {}
        target_v = {}
        for ti, tsp in enumerate(tsps):
            ninstrus_tsp = nsnds_tsp[ti+1]
            no_samples = ninstrus_tsp * (ninstrus_tsp - 1) / 2
            kernel = np.zeros((ninstrus_tsp, ninstrus_tsp))
            dkernel[tsp] = np.zeros((ninstrus_tsp, ninstrus_tsp, ndims))
            # offset = np.cumsum(nsnds_tsp[:ti+1])[-1]
            for i in range(ninstrus_tsp):
                for j in range(i + 1, ninstrus_tsp):
                    kernel[i, j] = -np.sum(
                        np.power(
                            np.divide(input_data[tsp][:, i] - input_data[tsp][:, j],
                                      (x + np.finfo(float).eps)), 2))
                    dkernel[tsp][i, j, :] = 2 * np.power((input_data[tsp][:, i] - input_data[tsp][:, j]), 2) / (np.power(x, 3) + np.finfo(float).eps)
            idx_triu = np.triu_indices(target_data[tsp].shape[0], k=1)
            target_v[tsp] = target_data[tsp][idx_triu]
            mean_target = np.mean(target_v[tsp])
            std_target = np.std(target_v[tsp])
            kernel_v[tsp] = kernel[idx_triu]
            mean_kernel = np.mean(kernel_v[tsp])
            std_kernel = np.std(kernel_v[tsp])
            Jn = np.sum(np.multiply(kernel_v[tsp] - mean_kernel, target_v[tsp] - mean_target))
            Jd = no_samples * std_target * std_kernel
            corr_sum += Jn/Jd
        for k in range(ndims):
            for ti, tsp in enumerate(tsps):
                idx_triu = np.triu_indices(target_data[tsp].shape[0], k=1)
                dkernel_k_v = dkernel[tsp][:, :, k][idx_triu]
                # print(tsp, dkernel_k_v.shape, target_v[tsp].shape, np.mean(target_v[tsp]))
                dJn = np.sum(dkernel_k_v * (target_v[tsp] - np.mean(target_v[tsp])))
                dJd = no_samples / no_samples * \
                            np.std(target_v[tsp]) / (np.std(kernel_v[tsp]) + np.finfo(float).eps) * \
                            np.sum(dkernel_k_v * (kernel_v[tsp] - np.mean(kernel_v[tsp])))
                gradients[k] += (Jd * dJn - Jn * dJd) / (np.power(Jd,2) + np.finfo(float).eps)
        return gradients


    def print_corr(xk):
        # kernel = np.zeros((ninstrus, ninstrus))
        # for i in range(ninstrus):
        #         for j in range(i + 1, ninstrus):
        #             kernel[i, j] = np.exp(-np.sum(
        #                 np.power(
        #                     np.divide(input_data[:, i] - input_data[:, j],
        #                               (xk + np.finfo(float).eps)), 2)))
        corr_sum = 0
        for ti, tsp in enumerate(tsps):
            # offset = np.cumsum(nsnds_tsp[:ti+1])[-1]
            ninstrus_tsp = nsnds_tsp[ti+1]
            no_samples = ninstrus_tsp * (ninstrus_tsp - 1) / 2
            kernel = np.zeros((ninstrus_tsp, ninstrus_tsp))
            for i in range(ninstrus_tsp):
                for j in range(i + 1, ninstrus_tsp):
                    kernel[i, j] = np.exp(-np.sum(
                        np.power(
                            np.divide(input_data[tsp][:, i] - input_data[tsp][:, j],
                                      (xk + np.finfo(float).eps)), 2)))
            idx_triu = np.triu_indices(target_data[tsp].shape[0], k=1)
            target_v = target_data[tsp][idx_triu]
            mean_target = np.mean(target_v)
            std_target = np.std(target_v)
            kernel_v = kernel[idx_triu]
            mean_kernel = np.mean(kernel_v)
            std_kernel = np.std(kernel_v)
            Jn = np.sum(np.multiply(kernel_v - mean_kernel, target_v - mean_target))
            Jd = no_samples * std_target * std_kernel
            corr_sum += (Jn/Jd) / len(tsps)
        if not os.path.isfile(os.path.join(log_foldername, 'tmp.pkl')):
            loop_cpt = 1
            pickle.dump({'loop': loop_cpt, 'correlation': [Jn/Jd]}, open(os.path.join(log_foldername, 'tmp.pkl'), 'wb'))
            correlations = [Jn/Jd]
            pickle.dump({
                    'sigmas': xk,
                    'kernel': kernel,
                    'Jn': Jn,
                    'Jd': Jd,
                    'correlations': correlations,
                }, open(os.path.join(log_foldername,
                    'optim_process_l={}.pkl'.format(loop_cpt)), 'wb'))
        else:
            last_loop = pickle.load(open(os.path.join(log_foldername,'tmp.pkl'), 'rb'))
            loop_cpt = last_loop['loop'] + 1
            correlations = last_loop['correlation']
            correlations.append(Jn/Jd)
            monitoring_step = 5
            if (loop_cpt % monitoring_step == 0):
                print('  |_ loop={} J={:.6f}'.format(loop_cpt, Jn/Jd))
                pickle.dump({
                    'sigmas': xk,
                    'kernel': kernel,
                    'Jn': Jn,
                    'Jd': Jd,
                    'correlations': correlations,
                }, open(os.path.join(log_foldername,
                    'optim_process_l={}.pkl'.format(loop_cpt)), 'wb'))
            pickle.dump({'loop': loop_cpt, 'correlation': correlations, 'sigmas': xk}, open(os.path.join(log_foldername, 'tmp.pkl'), 'wb'))

    res = minimize(corr, sigmas, args=(), method=method, jac=grad_corr, callback=print_corr, options=optim_options, bounds=optim_bounds)
    last_loop = pickle.load(open(os.path.join(log_foldername,'tmp.pkl'), 'rb'))
    sigmas_ = last_loop['sigmas']
    return correlations, sigmas_


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
