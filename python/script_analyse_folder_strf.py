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

timbrespace_db = load.database()
representations = [
    # 'fourier_spectrum',
    # 'fourier_spectrogram',
    'fourier_mps',
    'fourier_strf',
    'auditory_spectrum',
    'auditory_spectrogram',
    'auditory_mps',
    'auditory_strf'
]
# all_strfs = load_timbrespaces()
# all_dismatrices = load_dismatrices()

for i, tsp in enumerate(timbrespace_db.keys()):
    print('Processing', tsp)
    for rs in representations:
        print('Representation: {}'.format(rs))
        # compute/load strf for this particular tsp
        features = load.timbrespace_features(
            tsp,
            representations=[rs],
            window=None,
            timbrespace_db=None)
        strfs = features[rs]
        # get tje dissimilarity matrix associated
        dissimil_mat = load.timbrespace_dismatrix(tsp, timbrespace_db)
        # PCA
        tab_red = []
        rs_type = rs.split('_')[-1]
        if rs_type == 'strf':
            n_components = 1
            for i in range(len(strfs)):
                print('PCA on sound %02i' % (i + 1))
                strf_reduced = pca.pca(
                    np.absolute(strfs[i]),
                    strfs[i].shape[1],
                    n_components=n_components).flatten()
                tab_red.append(strf_reduced / np.max(strf_reduced))
            tab_red = np.transpose(np.asarray(tab_red))
        elif rs_type == 'spectrogram' or rs_type == 'mps':
            for i in range(len(strfs)):
                tab_red.append(strfs[i].flatten())
            tab_red = np.transpose(np.asarray(tab_red))
            # print(len(tab_red))
            print(tab_red.shape)
        elif rs_type == 'spectrum':
            for i in range(len(strfs)):
                tab_red.append(strfs[i])
            # 128 x nb sounds (time or freq?)
            tab_red = np.transpose(np.asarray(tab_red))
            print(tab_red.shape)

        # optimization arguments
        optim_args = {
            'cost': 'correlation',
            'init_sig_mean': 10.0,
            'init_sig_var': 1.0,
            'num_loops': 5000,
        }
        # optimization
        correlations, sigmas = training.kernel_optim(tab_red, dissimil_mat,
                                                     **optim_args)

        # # log results in txt files and arguments in pickle file
        log_filename = tsp.lower() + '_' + rs + '_' + time.strftime('%y%m%d-%H%M%S')
        np.savetxt('outs/' + log_filename + '_sigmas.txt', sigmas)
        np.savetxt('outs/' + log_filename + '_corrs.txt', correlations)
        # optim_args.update({'pca_components': n_components})
        pickle.dump(optim_args,
                    open('outs/' + log_filename + '_optim_args.pkl', 'wb'))
