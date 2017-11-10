# Copyright (c) Baptiste Caramiaux, Etienne Thoret
# All rights reserved

import numpy as np
import matplotlib.pylab as plt
import os
import pickle
import projection
import training
import utils
import datasets
import time


def analyze_specific(timbrespace_name='Iverson1993Whole'):
    test_args = {
        'window': None,
        'dimred_components': 1,
        'optim_args': {
            'cost': 'correlation',
            'init_sig_mean': 10.0,
            'init_sig_var': 0.5,
            'num_loops': 50000,
        }
    }
    timbrespace_dissimat = datasets.load_timbrespace_dismatrix(
        timbrespace_name)
    timbrespace_features = datasets.load_timbrespace_features(
        timbrespace_name, representations=['strf'], window=test_args['window'])
    features_reduced = [projection.pca(np.absolute(timbrespace_features['strf'][snd_i]), \
                                       timbrespace_features['strf'][snd_i].shape[1], \
                                       n_components=test_args['dimred_components']).flatten()
                            for snd_i in range(len(timbrespace_features['strf']))]
    features_reduced = np.transpose(np.asarray(features_reduced))
    # optimization
    correlations, sigmas = training.kernel_optim(
        features_reduced, timbrespace_dissimat, **test_args['optim_args'])
    # log results in txt files and arguments in pickle file
    log_filename = timbrespace_name.lower() + '_gbl-analysis_' + time.strftime(
        '%y%m%d-%H%M%S')
    np.savetxt('outs/' + log_filename + '_sigmas.txt', sigmas)
    np.savetxt('outs/' + log_filename + '_corrs.txt', correlations)
    pickle.dump(test_args, open('outs/' + log_filename + '_args.pkl','wb'))


def analyze_specific_windowed(timbrespace_name='Iverson1993Whole'):
    test_args = {
        'window': {
            'win_length': 0.2,
            'hop_length': 0.2
        },
        'dimred_components': 1,
        'optim_args': {
            'cost': 'correlation',
            'init_sig_mean': 10.0,
            'init_sig_var': 0.5,
            'num_loops': 50000,
        }
    }
    timbrespace_dissimat = datasets.load_timbrespace_dismatrix(
        timbrespace_name)
    # windowed case:
    timbrespace_features = datasets.load_timbrespace_features(
        timbrespace_name, representations=['strf'], window=test_args['window'])
    for select_frame in range(0, 1):
        features_reduced = [projection.pca(np.absolute(timbrespace_features['strf'][snd_i][select_frame]), \
                                           timbrespace_features['strf'][snd_i][select_frame].shape[1], \
                                           n_components=test_args['dimred_components']).flatten()
                                for snd_i in range(len(timbrespace_features['strf']))]
        features_reduced = np.transpose(np.asarray(features_reduced))
        # optimization
        correlations, sigmas = training.kernel_optim(
            features_reduced, timbrespace_dissimat, **test_args['optim_args'])
        # log results in txt files and arguments in pickle file
        log_filename = timbrespace_name.lower() + '_locanalysis-f' + str(
            select_frame) + '_' + time.strftime('%y%m%d-%H%M%S')
        np.savetxt('outs/' + log_filename + '_sigmas.txt', sigmas)
        np.savetxt('outs/' + log_filename + '_corrs.txt', correlations)
        pickle.dump(test_args, open('outs/' + log_filename + '_args.pkl','wb'))


def plots():
    correlations = {'gbl': [], 'loc': []}
    for root, dirs, files in os.walk('outs/'):
        for name in files:
            if(name.split('_')[1] == 'gbl-analysis'):
                if(name.split('_')[-1] == 'corrs.txt'):
                    correlations['gbl'].append(np.loadtxt('outs/'+name))
                    # plt.plot(correlations['gbl'][-1], 'k')
            if(name.split('_')[1] == 'locanalysis-f0'):
                if(name.split('_')[-1] == 'corrs.txt'):
                    correlations['loc'].append(np.loadtxt('outs/'+name))
    for k in correlations.keys():
        corrs = np.array(correlations[k])
        plt.plot(np.mean(corrs, axis=0))
    plt.show()
    



def analyze_all():

    timbrespace_db = load.load_timbrespace_database()
    # all_strfs = load_timbrespaces()
    # all_dismatrices = load_dismatrices()

    for i, tsp in enumerate(timbrespace_db.keys()):
        print('Processing', tsp)
        # compute/load strf for this particular tsp
        strfs = load.timbrespace_strf(tsp, timbrespace_db)
        # get tje dissimilarity matrix associated
        dissimil_mat = load.timbrespace_dismatrix(tsp, timbrespace_db)
        # PCA
        tab_red = []
        n_components = 1
        for i in range(len(strfs)):
            print('PCA on sound %02i' % (i + 1))
            strf_reduced = pca.pca(
                np.absolute(strfs[i]),
                strfs[i].shape[1],
                n_components=n_components).flatten()
            tab_red.append(strf_reduced / np.max(strf_reduced))
        tab_red = np.transpose(np.asarray(tab_red))

        # optimization arguments
        optim_args = {
            'cost': 'correlation',
            'init_sig_mean': 10.0,
            'init_sig_var': 1.0,
            'num_loops': 30000,
        }
        # optimization
        correlations, sigmas = kernel_optim(tab_red, dissimil_mat,
                                            **optim_args)

        # log results in txt files and arguments in pickle file
        log_filename = tsp.lower() + '_' + time.strftime('%y%m%d-%H%M%S')
        np.savetxt('outs/' + log_filename + '_sigmas.txt', sigmas)
        np.savetxt('outs/' + log_filename + '_corrs.txt', correlations)
        optim_args.update({'pca_components': n_components})
        pickle.dump(optim_args,
                    open('outs/' + log_filename + '_optim_args.pkl', 'wb'))


if __name__ == "__main__":
    plots()
