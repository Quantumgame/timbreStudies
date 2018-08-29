# Copyright (c) Baptiste Caramiaux, Etienne Thoret
# All rights reserved

import subprocess
import numpy as np
import os
import time
import pickle
import matplotlib.pylab as plt
from lib import pca
from lib import load
from lib import training
import random
from scipy.optimize import minimize


def compute_representations():
    args = {
        'timbre_spaces': list(sorted(load.database().keys())),
        'audio_representations': [
            'auditory_spectrum', 'fourier_spectrum', 'auditory_strf',
            'fourier_strf', 'auditory_spectrogram', 'fourier_spectrogram',
            'auditory_mps', 'fourier_mps'
        ],
        'log_foldername': './outs/ts_crossval',
        'audio_args': {
            'resampling_fs': 16000,
            'duration': 0.25,
            'duration_cut_decay': 0.05
        }
    }
    log_foldername = args['log_foldername']
    print('--processing')
    for i, tsp in enumerate(args['timbre_spaces']):
        print('Processing', tsp)
        subprocess.call(['mkdir', '-p', log_foldername+'/input_data'])
        for rs in args['audio_representations']:
            aud_repres = load.timbrespace_features(tsp, representations=[rs], audio_args=args['audio_args'])[rs]
            tab_red = []
            rs_type = rs.split('_')[-1]
            mapping = []
            variances = []
            if rs_type == 'strf':
                n_components = 1
                for i in range(len(aud_repres)):
                    strf_reduced, mapping_, variances = pca.pca_patil(
                        np.absolute(aud_repres[i]),
                        aud_repres[i].shape[1],
                        n_components=n_components)
                    strf_reduced = strf_reduced.flatten()
                    tab_red.append(strf_reduced)
                    mapping.append(mapping_)
                tab_red = np.transpose(np.asarray(tab_red))
            elif rs_type == 'spectrogram' or rs_type == 'mps':
                for i in range(len(aud_repres)):
                    tab_red.append(aud_repres[i].flatten())
                tab_red = np.transpose(np.asarray(tab_red))
            elif rs_type == 'spectrum':
                for i in range(len(aud_repres)):
                    tab_red.append(aud_repres[i])
                tab_red = np.transpose(np.asarray(tab_red))
            np.savetxt(log_foldername+'/input_data/{}_{}_input_data.txt'.format(tsp, rs), tab_red)
    print('--normalising')
    for rs in args['audio_representations']:
        print('Processing', rs)
        normas = []
        for i, tsp in enumerate(args['timbre_spaces']):
            data = np.loadtxt(log_foldername+'/input_data/{}_{}_input_data.txt'.format(tsp, rs))
            normas.append(np.mean(np.max(np.abs(data), axis=0)))
        for i, tsp in enumerate(args['timbre_spaces']):
            data = np.loadtxt(log_foldername+'/input_data/{}_{}_input_data.txt'.format(tsp, rs))
            data = data / np.mean(normas)
            # print('  save final data {} {:.5f}'.format(data.shape, np.mean(normas)))
            np.savetxt(log_foldername+'/input_data/{}_{}_input_data.txt'.format(tsp, rs), data)
            # data = data / np.mean(np.max(np.abs(data), axis=0))
            # np.savetxt(log_foldername+'/input_data/{}_{}_input_data.txt'.format(tsp, rs), data)



def run_crossts(rs, args):
    rslog_foldername = args['optim_args']['log_foldername']
    rs = rslog_foldername.split('/')[-1].split('-')[0]
    print('Audio representation:', rs)

    cost=args['optim_args']['cost']
    loss=args['optim_args']['loss']
    init_sig_mean=args['optim_args']['init_sig_mean']
    init_sig_var=args['optim_args']['init_sig_var']
    num_loops=args['optim_args']['num_loops']
    method='L-BFGS-B'
    log_foldername=rslog_foldername
    logging=args['optim_args']['logging']
    verbose=True

    tsp  = args['timbre_spaces'][0]
    dissimil_mat_tsp = load.timbrespace_dismatrix(tsp, load.database())
    print(log_foldername+'/../input_data/{}_{}_input_data.txt'.format(tsp, rs))
    tab_red = np.loadtxt(log_foldername+'/../input_data/{}_{}_input_data.txt'.format(tsp, rs))

    ndims = tab_red.shape[0]

    sigmas = np.abs(init_sig_mean + init_sig_var * np.random.randn(ndims, 1))
    init_seed = sigmas
    
    correlations = []  # = np.zeros((num_loops, ))
    retrieved_loop = 0


    # if (verbose):
    #     print("* training sigmas of gaussian kernels with cost '{}' and method '{}'".format(
    #         cost, method))
    

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
        corr_sum = 0
        for tsp in args['timbre_spaces']:
            dissimil_mat_tsp = load.timbrespace_dismatrix(tsp, load.database())
            tab_red = np.loadtxt(log_foldername+'/../input_data/{}_{}_input_data.txt'.format(tsp, rs))
            # tab_red = np.loadtxt('input_data/{}_{}_input_data.txt'.format(tsp, rs))
            
            ndims, ninstrus = tab_red.shape[0], tab_red.shape[1]
            no_samples = ninstrus * (ninstrus - 1) / 2
            idx_triu = np.triu_indices(dissimil_mat_tsp.shape[0], k=1)
            target_v = dissimil_mat_tsp[idx_triu]
            mean_target = np.mean(target_v)
            std_target = np.std(target_v)

            kernel = np.zeros((ninstrus, ninstrus))
            for i in range(ninstrus):
                    for j in range(i + 1, ninstrus):
                        kernel[i, j] = -np.sum(
                            np.power(
                                np.divide(tab_red[:, i] - tab_red[:, j],
                                          (x + np.finfo(float).eps)), 2))
            kernel_v = kernel[idx_triu]
            mean_kernel = np.mean(kernel_v)
            std_kernel = np.std(kernel_v)
            Jn = np.sum(np.multiply(kernel_v - mean_kernel, target_v - mean_target))
            Jd = no_samples * std_target * std_kernel
            corr_sum += Jn/Jd
        return corr_sum 


    def grad_corr(x):
        corr_sum = 0
        for tsp_i, tsp in enumerate(args['timbre_spaces']):
            dissimil_mat_tsp = load.timbrespace_dismatrix(tsp, load.database())
            tab_red = np.loadtxt(log_foldername+'/../input_data/{}_{}_input_data.txt'.format(tsp, rs)) #np.loadtxt('input_data/{}_{}_input_data.txt'.format(tsp, rs))
            # tab_red = np.loadtxt('input_data/{}_{}_input_data.txt'.format(tsp, rs))

            ndims, ninstrus = tab_red.shape[0], tab_red.shape[1]
            no_samples = ninstrus * (ninstrus - 1) / 2
            idx_triu = np.triu_indices(dissimil_mat_tsp.shape[0], k=1)
            target_v = dissimil_mat_tsp[idx_triu]
            mean_target = np.mean(target_v)
            std_target = np.std(target_v)

            if tsp_i==0:
                gradients = np.zeros((ndims, 1))

            kernel = np.zeros((ninstrus, ninstrus))
            dkernel = np.zeros((ninstrus, ninstrus, ndims))
            for i in range(ninstrus):
                    for j in range(i + 1, ninstrus):
                        kernel[i, j] = -np.sum(
                            np.power(
                                np.divide(tab_red[:, i] - tab_red[:, j],
                                          (x + np.finfo(float).eps)), 2))
                        dkernel[i, j, :] = 2 * np.power((tab_red[:, i] - tab_red[:, j]), 2) / (np.power(x, 3) + np.finfo(float).eps)
            kernel_v = kernel[idx_triu]
            mean_kernel = np.mean(kernel_v)
            std_kernel = np.std(kernel_v)
            Jn = np.sum(np.multiply(kernel_v - mean_kernel, target_v - mean_target))
            Jd = no_samples * std_target * std_kernel

            for k in range(ndims):
                dkernel_k_v = dkernel[:, :, k][idx_triu]
                dJn = np.sum(dkernel_k_v * (target_v - mean_target))
                dJd = no_samples / no_samples * \
                            std_target / (std_kernel + np.finfo(float).eps) * \
                            np.sum(dkernel_k_v * (kernel_v - mean_kernel))
                gradients[k] += (Jd * dJn - Jn * dJd) / (np.power(Jd,2) + np.finfo(float).eps)
        return gradients


    def print_corr(xk):
        jns, jds = [], []
        corr_sum = []
        for tsp in args['timbre_spaces']:
            dissimil_mat_tsp = load.timbrespace_dismatrix(tsp, load.database())
            tab_red = np.loadtxt(log_foldername+'/../input_data/{}_{}_input_data.txt'.format(tsp, rs))
            
            ndims, ninstrus = tab_red.shape[0], tab_red.shape[1]
            no_samples = ninstrus * (ninstrus - 1) / 2
            idx_triu = np.triu_indices(dissimil_mat_tsp.shape[0], k=1)
            target_v = dissimil_mat_tsp[idx_triu]
            mean_target = np.mean(target_v)
            std_target = np.std(target_v)

            kernel = np.zeros((ninstrus, ninstrus))
            for i in range(ninstrus):
                    for j in range(i + 1, ninstrus):
                        kernel[i, j] = -np.sum(
                            np.power(
                                np.divide(tab_red[:, i] - tab_red[:, j],
                                          (xk + np.finfo(float).eps)), 2))
            kernel_v = kernel[idx_triu]
            mean_kernel = np.mean(kernel_v)
            std_kernel = np.std(kernel_v)
            Jn = np.sum(np.multiply(kernel_v - mean_kernel, target_v - mean_target))
            Jd = no_samples * std_target * std_kernel
            corr_sum.append(Jn/Jd)
        
        if not os.path.isfile(os.path.join(log_foldername, 'tmp.pkl')):
            loop_cpt = 1
            pickle.dump({'loop': loop_cpt, 'correlation': [np.mean(corr_sum)]}, open(os.path.join(log_foldername, 'tmp.pkl'), 'wb'))
            correlations = [np.mean(corr_sum)]
            pickle.dump({
                    'sigmas': xk,
                    'kernel': kernel,
                    'Jn': Jn,
                    'Jd': Jd,
                    'corr_sum': corr_sum,
                    'correlations': correlations,
                }, open(os.path.join(log_foldername,
                    'optim_process_l={}.pkl'.format(loop_cpt)), 'wb'))
        else:
            last_loop = pickle.load(open(os.path.join(log_foldername,'tmp.pkl'), 'rb'))
            loop_cpt = last_loop['loop'] + 1
            correlations = last_loop['correlation']
            correlations.append(np.mean(corr_sum))
            monitoring_step = 50
            if (loop_cpt % monitoring_step == 0):
                corr_sum_str = ' '.join(['{:.2f}'.format(c) for c in corr_sum])
                print('  |_ loop={} J={:.6f} ({})'.format(loop_cpt, np.mean(corr_sum), corr_sum_str))
                pickle.dump({
                    'sigmas': xk,
                    'kernel': kernel,
                    'Jn': Jn,
                    'Jd': Jd,
                    'corr_sum': corr_sum,
                    'correlations': correlations,
                }, open(os.path.join(log_foldername,
                    'optim_process_l={}.pkl'.format(loop_cpt)), 'wb'))
            pickle.dump({'loop': loop_cpt, 'correlation': correlations, 'sigmas': xk}, open(os.path.join(log_foldername, 'tmp.pkl'), 'wb'))

    res = minimize(corr, sigmas, args=(), method=method, jac=grad_corr, callback=print_corr, options=optim_options, bounds=optim_bounds)
    last_loop = pickle.load(open(os.path.join(log_foldername,'tmp.pkl'), 'rb'))
    sigmas_ = last_loop['sigmas']
    return correlations, sigmas_


def run_optimization(args={}):
    log_foldername = args['log_foldername']
    # log_foldername = os.path.join(orig_fn, 'crossts')
    subprocess.call(['mkdir', '-p', log_foldername])
    for rs in args['audio_representations']:
        rslog_foldername = log_foldername + '/' + rs + '-' + time.strftime('%y%m%d@%H%M%S')
        subprocess.call(['mkdir', '-p', rslog_foldername])
        args['optim_args'].update({'log_foldername': rslog_foldername})
        run_crossts(rs, args)


def run_optim_tests():
    args = {
        'timbre_spaces': list(sorted(load.database().keys())),
        'audio_representations': [
            'auditory_spectrum', 'fourier_spectrum', 'auditory_strf',
            'fourier_strf', 'auditory_spectrogram', 'fourier_spectrogram',
            'auditory_mps', 'fourier_mps'
        ],
        'log_foldername': './outs/ts_crossval',
        'audio_args': {
            'resampling_fs': 16000,
            'duration': 0.25,
            'duration_cut_decay': 0.05
        },
        'optim_args': {
            'cost': 'correlation',
            'loss': 'exp_sum',
            'method': 'L-BFGS-B',
            'init_sig_mean': 1.0,
            'init_sig_var': 0.01,
            'num_loops': 1000,
            'logging': True
        },
    }
    start_time = time.time()
    run_optimization(args)
    print(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


def compute_correlations_from_within_ts_tests():
    args = {
        'timbre_spaces': list(sorted(load.database().keys())),
        'audio_representations': [
            'auditory_spectrum', 'fourier_spectrum', 'auditory_strf',
            'fourier_strf', 'auditory_spectrogram', 'fourier_spectrogram',
            'auditory_mps', 'fourier_mps'
        ],
        'res_foldername': './results_07-24-2018',
        'log_foldername': './outs/ts_crossval_fixed',
    }
    corrs = []
    for rs in args['audio_representations']:
        print(rs)
        corrs_rs = []
        for tsp_i, tsp in enumerate(args['timbre_spaces']):
            print(' ',tsp)
            input_data = np.loadtxt(args['res_foldername']+'/{}_{}_input_data.txt'.format(tsp, rs))
            target_data = np.loadtxt(args['res_foldername']+'/{}_{}_target_data.txt'.format(tsp, rs))
            sigmas_ref = np.loadtxt(args['res_foldername']+'/{}_{}_sigmas.txt'.format(tsp, rs))
            corrs_rs_tsp = []
            all_corrc = []
            all_sigmas = []
            for tsp_j, tsp_2 in enumerate(args['timbre_spaces']):
                if (tsp_i != tsp_j):
                    sigmas = np.loadtxt(args['res_foldername']+'/{}_{}_sigmas.txt'.format(tsp_2, rs))
                    all_sigmas.append(sigmas)
            sigmas = np.mean(all_sigmas, axis=0)

            for tsp_j, tsp_2 in enumerate(args['timbre_spaces']):
                if (tsp_i != tsp_j):
                    ndims, ninstrus = input_data.shape[0], input_data.shape[1]
                    no_samples = ninstrus * (ninstrus - 1) / 2
                    idx_triu = np.triu_indices(target_data.shape[0], k=1)
                    target_v = target_data[idx_triu]
                    mean_target = np.mean(target_v)
                    std_target = np.std(target_v)
                    kernel = np.zeros((ninstrus, ninstrus))
                    for i in range(ninstrus):
                            for j in range(i + 1, ninstrus):
                                kernel[i, j] = -np.sum(
                                    np.power(
                                        np.divide(input_data[:, i] - input_data[:, j],
                                                  (sigmas + np.finfo(float).eps)), 2))
                    kernel_v = kernel[idx_triu]
                    mean_kernel = np.mean(kernel_v)
                    std_kernel = np.std(kernel_v)
                    Jn = np.sum(np.multiply(kernel_v - mean_kernel, target_v - mean_target))
                    Jd = no_samples * std_target * std_kernel
                    corrs_rs_tsp.append(Jn/Jd)

            # for tsp_j, tsp_2 in enumerate(args['timbre_spaces']):
            #     if (tsp_i != tsp_j):
            #         sigmas = np.loadtxt(args['res_foldername']+'/{}_{}_sigmas.txt'.format(tsp_2, rs))
            #         ndims, ninstrus = input_data.shape[0], input_data.shape[1]
            #         no_samples = ninstrus * (ninstrus - 1) / 2
            #         idx_triu = np.triu_indices(target_data.shape[0], k=1)
            #         target_v = target_data[idx_triu]
            #         mean_target = np.mean(target_v)
            #         std_target = np.std(target_v)
            #         kernel = np.zeros((ninstrus, ninstrus))
            #         for i in range(ninstrus):
            #                 for j in range(i + 1, ninstrus):
            #                     kernel[i, j] = -np.sum(
            #                         np.power(
            #                             np.divide(input_data[:, i] - input_data[:, j],
            #                                       (sigmas + np.finfo(float).eps)), 2))
            #         kernel_v = kernel[idx_triu]
            #         mean_kernel = np.mean(kernel_v)
            #         std_kernel = np.std(kernel_v)
            #         Jn = np.sum(np.multiply(kernel_v - mean_kernel, target_v - mean_target))
            #         Jd = no_samples * std_target * std_kernel
            #         corrs_rs_tsp.append(Jn/Jd)
            #         print('   w/ {}: {:.5f} ({:.4f}, {:.4f})'.format(tsp_2, Jn/Jd, Jn, Jd))
            #         corrc = np.corrcoef(sigmas_ref, sigmas)[0,1]
            #         all_sigmas.append(sigmas)
            #         all_corrc.append(corrc)
            # print(' --- {}', np.mean(all_corrc))
            corrs_rs.append(np.mean(corrs_rs_tsp))
        corrs.append(corrs_rs)
    corrs_m = [np.mean(corrs_rs) for corrs_rs in corrs]
    idx = sorted(range(len(args['audio_representations'])), key=lambda k: args['audio_representations'][k])
    sorted_corrs = [corrs_m[k] for k in idx]
    sorted_labels = [args['audio_representations'][k] for k in idx]
    x = np.arange(len(corrs_m))
    plt.figure(figsize=(12,8))
    plt.plot(sorted_corrs, '-ok')
    plt.xticks(x, sorted_labels)
    plt.ylabel('correlation')
    # plt.savefig(
    #     'correlation_cross_ts.pdf',
    #     bbox_inches='tight')
    plt.show()





if __name__ == '__main__':    
    # compute_representations()
    run_optim_tests()
    # compute_correlations_from_within_ts_tests()
    
    