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


def run_one_tsp_rs(tsp, rs, args):
    rslog_foldername = args['optim_args']['log_foldername']
    rs = rslog_foldername.split('/')[-1].split('-')[0]
    dissimil_mat = load.timbrespace_dismatrix(tsp, load.database())
    aud_repres = load.timbrespace_features(
        tsp, representations=[rs], audio_args=args['audio_args'])[rs]
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
    pickle.dump({
        'data_repres': aud_repres,
        'data_proj': tab_red,
        'mapping': mapping,
        'variances': variances,
        'dissimilarities': dissimil_mat,
        'audio_args': args['audio_args']
    }, open(os.path.join(rslog_foldername, 'dataset.pkl'), 'wb'))
    print('  data dimension:', tab_red.shape)
    print('* normalizing')
    tab_red = tab_red / np.mean(np.max(np.abs(tab_red), axis=0))
    correlations, _ = training.kernel_optim_lbfgs_log(tab_red, dissimil_mat,
                                               **args['optim_args'])


def run_one_tsp_rs_crossval(tsp, rs, args):
    rslog_foldername = args['optim_args']['log_foldername']
    rs = rslog_foldername.split('/')[-1].split('-')[0]
    dissimil_mat = load.timbrespace_dismatrix(tsp, load.database())
    aud_repres = load.timbrespace_features(
        tsp, representations=[rs], audio_args=args['audio_args'])[rs]
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
    pickle.dump({
        'data_repres': aud_repres,
        'data_proj': tab_red,
        'mapping': mapping,
        'variances': variances,
        'dissimilarities': dissimil_mat,
        'audio_args': args['audio_args']
    }, open(os.path.join(rslog_foldername, 'dataset.pkl'), 'wb'))
    print('  data dimension:', tab_red.shape)
    print('* normalizing')
    tab_red = tab_red / np.mean(np.max(np.abs(tab_red), axis=0))

    corr_fold = []
    sigmas_fold = []
    print('* cross-validation tests')
    for fold in range(20):

        idx = [i for i in range(tab_red.shape[1])]
        random.shuffle(idx)
        train_idx = idx[:int(2*len(idx)/3)]
        test_idx = idx[int(2*len(idx)/3):]
        dmat = dissimil_mat[train_idx, :]
        dmat = dmat[:, train_idx]
        print('* Fold {} - {} {}'.format(fold+1, train_idx, test_idx))
        correlations, sigmas = training.kernel_optim_lbfgs_log(tab_red[:,train_idx], dmat, **args['optim_args'])

        # testing
        test_data = tab_red[:,test_idx]
        dmat = dissimil_mat[test_idx, :]
        target_test_data = dmat[:, test_idx]
        idx_triu = np.triu_indices(target_test_data.shape[0], k=1)
        target_test_data_v = target_test_data[idx_triu]
        mean_target_test = np.mean(target_test_data_v)
        std_target_test = np.std(target_test_data_v)
        ninstrus = len(test_idx)
        no_samples = ninstrus * (ninstrus - 1) / 2
        kernel = np.zeros((ninstrus, ninstrus))
        for i in range(ninstrus):
            for j in range(i + 1, ninstrus):
                kernel[i, j] = -np.sum(np.power(
                                    np.divide(test_data[:, i] - test_data[:, j],
                                            (sigmas + np.finfo(float).eps)), 2))
        kernel_v = kernel[idx_triu]
        mean_kernel = np.mean(kernel_v)
        std_kernel = np.std(kernel_v)
        Jn = np.sum(np.multiply(kernel_v - mean_kernel, target_test_data_v - mean_target_test))
        Jd = no_samples * std_target_test * std_kernel
        corr_fold.append(Jn/Jd)
        sigmas_fold.append(sigmas)
        print('\tfold={} corr={}'.format(fold+1, Jn/Jd))
        pickle.dump({
            'corr_fold': corr_fold,
            'sigmas_fold': sigmas_fold
            }, open(os.path.join(rslog_foldername, 'crossval_res.pkl'), 'wb'))
    print('\tall folds: corr={} ({})'.format(np.mean(corr_fold), np.std(corr_fold)))


def run_optimization(args={}):
    orig_fn = args['log_foldername']
    for i, tsp in enumerate(args['timbre_spaces']):
        print('Processing', tsp)
        log_foldername = os.path.join(orig_fn, tsp.lower())
        subprocess.call(['mkdir', '-p', log_foldername])
        for rs in args['audio_representations']:
            rslog_foldername = log_foldername + '/' + rs + '-' + time.strftime(
                '%y%m%d@%H%M%S')
            subprocess.call(['mkdir', '-p', rslog_foldername])
            args['optim_args'].update({'log_foldername': rslog_foldername})
            if not args['test_args']['snd_crossval']:
                run_one_tsp_rs(tsp, rs, args)
            else:
                run_one_tsp_rs_crossval(tsp, rs, args)


def run(tsp=None, rs=None):
    valid_tsps = list(sorted(load.database().keys())) if tsp == None else tsp
    valid_rs = [
            'auditory_spectrum', 'fourier_spectrum', 'auditory_strf',
            'fourier_strf', 'auditory_spectrogram', 'fourier_spectrogram',
            'auditory_mps', 'fourier_mps'
        ] if rs == None else rs
    
    args = {
        'timbre_spaces': valid_tsps,
        'audio_representations': valid_rs,
        'log_foldername': './outs/',
        'test_args': {
            'snd_crossval': False
        },
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
            'num_loops': 300,
            'logging': True
        },
    }
    start_time = time.time()
    run_optimization(args)
    print(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))




# def run_test_durations():
#     args = {
#         'timbre_spaces': list(sorted(load.database().keys())),
#         'audio_representations': [
#             'auditory_spectrum', 'fourier_spectrum', 'auditory_strf',
#             'fourier_strf', 'auditory_spectrogram', 'fourier_spectrogram',
#             'auditory_mps', 'fourier_mps'
#         ],
#         'log_foldername':
#         '/Users/baptistecaramiaux/Work/Projects/TimbreProject_Thoret/Code/timbreStudies/python/tests/durations_logloss',
#         'audio_args': {
#             'resampling_fs': 16000,
#             'duration': 0.25,
#             'duration_cut_decay': 0.05,
#             'offset': 0.0
#         },
#         'optim_args': {
#             'cost': 'correlation',
#             'loss': 'sum',
#             'method': 'L-BFGS-B',
#             'init_sig_mean': 1.0,
#             'init_sig_var': 0.01,
#             'num_loops': 500,
#             'logging': True
#         },
#     }
#     the_path = args['log_foldername']
#     for duration in [0.25, 0.20, 0.15, 0.10]:
#         print('*** TEST DURATION', duration)
#         args['audio_args']['duration'] = duration
#         args['log_foldername'] = the_path + '/duration={:.2f}'.format(duration)
#         run_optimization(args)


# def run_test_offsets():
#     args = {
#         'timbre_spaces': list(sorted(load.database().keys())),
#         'audio_representations': [
#             'auditory_spectrum', 'fourier_spectrum', 'auditory_strf',
#             'fourier_strf', 'auditory_spectrogram', 'fourier_spectrogram',
#             'auditory_mps', 'fourier_mps'
#         ],
#         'log_foldername':
#         '/Users/baptistecaramiaux/Work/Projects/TimbreProject_Thoret/Code/timbreStudies/python/tests/offsets',
#         'audio_args': {
#             'resampling_fs': 16000,
#             'duration': 0.15,
#             'duration_cut_decay': 0.025,
#             'offset': 0.0
#         },
#         'optim_args': {
#             'cost': 'correlation',
#             'loss': 'sum',
#             'method': 'L-BFGS-B',
#             'init_sig_mean': 1.0,
#             'init_sig_var': 0.01,
#             'num_loops': 500,
#             'logging': True
#         },
#     }
#     the_path = args['log_foldername']
#     for offset in [0.10, 0.08, 0.06, 0.04, 0.02]:
#         print('\n*** TEST OFFSET', offset)
#         args['audio_args']['offset'] = offset
#         args['log_foldername'] = the_path + '/offset={:.2f}'.format(offset)
#         run_optimization(args)


# def run_fixes_test_durations():
#     args = {
#         'timbre_spaces': ['Grey1977'],
#         # 'audio_representations': [
#         #     'auditory_spectrum', 'fourier_spectrum', 'auditory_strf',
#         #     'fourier_strf', 'auditory_spectrogram', 'fourier_spectrogram',
#         #     'auditory_mps', 'fourier_mps'
#         # ],
#         'audio_representations': [
#             'fourier_spectrogram'
#         ],
#         'log_foldername':
#         '/Users/baptistecaramiaux/Work/Projects/TimbreProject_Thoret/Code/timbreStudies/python/tests/durations',
#         'audio_args': {
#             'resampling_fs': 16000,
#             'duration': 0.25,
#             'duration_cut_decay': 0.05
#         },
#         'optim_args': {
#             'cost': 'correlation',
#             'loss': 'exp_sum',
#             'method': 'L-BFGS-B',
#             'init_sig_mean': 1.0,
#             'init_sig_var': 0.01,
#             'num_loops': 1000,
#             'logging': True
#         },
#     }
#     for duration in [0.25]: # , 0.20, 0.15, 0.10]:
#         print('*** TEST DURATION', duration)
#         args['audio_args']['duration'] = duration
#         run_optimization(args)


if __name__ == '__main__':
    # run on all TSPs and RS: uncomment line below
    # run()
    # run on a single TSP and a single RS:
    run(tsp=['Grey1977'], rs=['auditory_spectrum']) 