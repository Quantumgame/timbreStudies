# Copyright (c) Baptiste Caramiaux, Etienne Thoret
# All rights reserved
import numpy as np
import matplotlib.pylab as plt
import os
import pickle
import pca
# from optimization import kernel_optim
from lib import load
from lib import training
import time
import subprocess

timbrespace_db = load.database()
representations = [
    # 'auditory_spectrum',
    'fourier_spectrum',
    'auditory_strf',
    'fourier_strf',
    # 'auditory_spectrogram', 
    # 'fourier_spectrogram',
    # 'auditory_mps', 
    # 'fourier_mps',
]


def run_once(tsp, rs, optim_args):
    rslog_foldername = optim_args['log_foldername']
    rs = rslog_foldername.split('/')[-1].split('-')[0]
    dissimil_mat = load.timbrespace_dismatrix(tsp, timbrespace_db)
    aud_repres = load.timbrespace_features(
        tsp,
        representations=[rs],
        window=None,
        timbrespace_db=None,
        verbose=True)[rs]
    tab_red = []
    rs_type = rs.split('_')[-1]
    mapping = []
    variances = []
    if rs_type == 'strf':
        n_components = 1
        for i in range(len(aud_repres)):
            # print('PCA on sound %02i' % (i + 1))
            strf_reduced, mapping, variances = pca.pca(
                np.absolute(aud_repres[i]),
                aud_repres[i].shape[1],
                n_components=n_components)
            strf_reduced = strf_reduced.flatten()
            tab_red.append(strf_reduced / np.max(strf_reduced))
        tab_red = np.transpose(np.asarray(tab_red))
    elif rs_type == 'spectrogram' or rs_type == 'mps':
        for i in range(len(aud_repres)):
            tab_red.append(aud_repres[i].flatten())
        tab_red = np.transpose(np.asarray(tab_red))
    elif rs_type == 'spectrum':
        for i in range(len(aud_repres)):
            tab_red.append(aud_repres[i])
        # 128 x nb sounds (time or freq?)
        tab_red = np.transpose(np.asarray(tab_red))
    pickle.dump({
        'data_repres': aud_repres,
        'data_proj': tab_red,
        'mapping': mapping,
        'variances': variances,
        'dissimilarities': dissimil_mat,
    }, open(os.path.join(rslog_foldername, 'dataset.pkl'), 'wb'))
    print('  data dimension:', tab_red.shape)
    print('* normalizing')
    tab_red = tab_red / np.max(np.max(np.abs(tab_red), axis=0))
    # plt.plot(tab_red)
    # plt.show()
    # optimization
    correlations = training.kernel_optim(tab_red, dissimil_mat, **optim_args)


def run_optimization(optim_args={}):
    for i, tsp in enumerate(sorted(timbrespace_db.keys())):
        print('Processing', tsp)
        # log_foldername = 'outs/' + tsp.lower() + '-' + time.strftime(
        #     '%y%m%d@%H%M%S')
        log_foldername = 'outs_all/' + tsp.lower()
        subprocess.call(['mkdir', '-p', log_foldername])
        for rs in representations:
            rslog_foldername = log_foldername + '/' + rs + '-' + time.strftime(
                '%y%m%d@%H%M%S')
            subprocess.call(['mkdir', '-p', rslog_foldername])
            optim_args['log_foldername'] = rslog_foldername
            run_once(tsp, rs, optim_args)
            # aud_repres = load.timbrespace_features(
            #     tsp,
            #     representations=[rs],
            #     window=None,
            #     timbrespace_db=None,
            #     verbose=True)[rs]
            # tab_red = []
            # rs_type = rs.split('_')[-1]
            # mapping = []
            # variances = []
            # if rs_type == 'strf':
            #     n_components = 1
            #     for i in range(len(aud_repres)):
            #         # print('PCA on sound %02i' % (i + 1))
            #         strf_reduced, mapping, variances = pca.pca(
            #             np.absolute(aud_repres[i]),
            #             aud_repres[i].shape[1],
            #             n_components=n_components)
            #         strf_reduced = strf_reduced.flatten()
            #         tab_red.append(strf_reduced / np.max(strf_reduced))
            #     tab_red = np.transpose(np.asarray(tab_red))
            # elif rs_type == 'spectrogram' or rs_type == 'mps':
            #     for i in range(len(aud_repres)):
            #         tab_red.append(aud_repres[i].flatten())
            #     tab_red = np.transpose(np.asarray(tab_red))
            # elif rs_type == 'spectrum':
            #     for i in range(len(aud_repres)):
            #         tab_red.append(aud_repres[i])
            #     # 128 x nb sounds (time or freq?)
            #     tab_red = np.transpose(np.asarray(tab_red))
            # pickle.dump({
            #     'data_repres': aud_repres,
            #     'data_proj': tab_red,
            #     'mapping': mapping,
            #     'variances': variances,
            #     'dissimilarities': dissimil_mat,
            # }, open(os.path.join(rslog_foldername, 'dataset.pkl'), 'wb'))
            # print('  data dimension:', tab_red.shape)
            # print('* normalizing')
            # tab_red = tab_red / np.mean(np.max(tab_red, axis=0))
            # # optimization arguments
            # optim_args = {
            #     'cost': 'correlation',
            #     'init_sig_mean': 10.0,
            #     'init_sig_var': 1.0,
            #     'num_loops': 30000,
            #     'log_foldername': rslog_foldername,
            #     'logging': True
            # }
            # # optimization
            # correlations = training.kernel_optim(
            #     tab_red, dissimil_mat, **optim_args)


def run_all():
    optim_args = {
        'cost': 'correlation',
        'init_sig_mean': 1.0,
        'init_sig_var': 0.01,
        'num_loops': 100000,
        'learning_rate': 0.1,
        'log_foldername': './',
        'logging': True
    }
    run_optimization(optim_args)


def grid_search_lr():
    for learning_rate in [1.0, 0.1, 0.01, 0.001]:
        for n_test in range(5):
            print('***', learning_rate, n_test + 1)
            optim_args = {
                'cost': 'correlation',
                'init_sig_mean': 1.0,
                'init_sig_var': 0.01,
                'num_loops': 40000,
                'learning_rate': learning_rate,
                'log_foldername': './',
                'logging': True
            }
            run_optimization(optim_args)


def resume_all(resumefn='./outs/'):
    for i, tsp in enumerate(sorted(timbrespace_db.keys())):
        print('Processing', tsp)
        dissimil_mat = load.timbrespace_dismatrix(tsp, timbrespace_db)
        for rs in representations:
            # for el in dir_names:
            #     if el[1].split('-')[0] == tsp.lower() and el[1].split(
            #             '-')[1] == rs:
            rslog_foldername = './outs/' + tsp.lower() + '/' + rs
            if os.path.isdir(rslog_foldername):
                # retrieve_foldername = os.path.join(el[0], el[1])
                training.resume_kernel_optim(
                    rslog_foldername,
                    rslog_foldername,
                    num_loops=100000,
                    logging=True)
            else:
                subprocess.call(['mkdir', '-p', rslog_foldername])
                run_once(tsp, rs, rslog_foldername)


def nonopt_correlations():
    corr_results = {}
    for i, tsp in enumerate(timbrespace_db.keys()):
        print('Processing', tsp)
        corr_results[tsp] = {}
        target_data = load.timbrespace_dismatrix(tsp, timbrespace_db)
        for rs in sorted(representations):
            aud_repres = load.timbrespace_features(
                tsp,
                representations=[rs],
                window=None,
                timbrespace_db=None,
                verbose=False)[rs]
            tab_red = []
            rs_type = rs.split('_')[-1]
            if rs_type == 'strf':
                n_components = 1
                for i in range(len(aud_repres)):
                    # print('PCA on sound %02i' % (i + 1))
                    strf_reduced = pca.pca(
                        np.absolute(aud_repres[i]),
                        aud_repres[i].shape[1],
                        n_components=n_components).flatten()
                    tab_red.append(strf_reduced / np.max(strf_reduced))
                tab_red = np.transpose(np.asarray(tab_red))
            elif rs_type == 'spectrogram' or rs_type == 'mps':
                for i in range(len(aud_repres)):
                    tab_red.append(aud_repres[i].flatten())
                tab_red = np.transpose(np.asarray(tab_red))
            elif rs_type == 'spectrum':
                for i in range(len(aud_repres)):
                    tab_red.append(aud_repres[i])
                # 128 x nb sounds (time or freq?)
                tab_red = np.transpose(np.asarray(tab_red))
            input_data = tab_red / np.mean(np.std(tab_red, axis=0))

            # plt.plot(input_data)
            # plt.show()
            ndims, ninstrus = input_data.shape[0], input_data.shape[1]
            no_samples = ninstrus * (ninstrus - 1) / 2
            idx_triu = np.triu_indices(target_data.shape[0], k=1)
            target_v = target_data[idx_triu]
            mean_target = np.mean(target_v)
            std_target = np.std(target_v)
            kernel = np.zeros((ninstrus, ninstrus))
            for i in range(ninstrus):
                for j in range(i + 1, ninstrus):
                    kernel[i, j] = np.sum(
                        np.power(input_data[:, i] - input_data[:, j], 2))
            kernel_v = kernel[idx_triu]
            mean_kernel = np.mean(kernel_v)
            std_kernel = np.std(kernel_v)
            Jn = np.sum(
                np.multiply(kernel_v - mean_kernel, target_v - mean_target))
            Jd = (no_samples - 1) * std_target * std_kernel
            corr_results[tsp][rs] = Jn / Jd
            print('  {} : {}'.format(rs, Jn / Jd))
    pickle.dump(corr_results, open('correlations_results.pkl', 'wb'))


if __name__ == '__main__':
    # resume_all()
    # run_all()
    # nonopt_correlations()
    # run_optimization()
    # resume_all()
    grid_search_lr()
