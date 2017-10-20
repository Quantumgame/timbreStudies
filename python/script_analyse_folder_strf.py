# Copyright (c) Baptiste Caramiaux, Etienne Thoret
# All rights reserved

import numpy as np
import matplotlib.pylab as plt
import os
import pickle
import pca
from features import STRF
from optimization import kernel_optim
import utils
import load
import time


timbrespace_db = load.load_timbrespace_database()
# all_strfs = load_timbrespaces()
# all_dismatrices = load_dismatrices()

for i, tsp in enumerate(timbrespace_db.keys()):
    print('Processing',tsp)
    # compute/load strf for this particular tsp
    strfs = load.timbrespace_strf(tsp, timbrespace_db)
    # get tje dissimilarity matrix associated
    dissimil_mat = load.timbrespace_dismatrix(tsp, timbrespace_db)
    # PCA
    tab_red = []
    n_components = 1
    for i in range(len(strfs)):
        print('PCA on sound %02i' % (i + 1))
        strf_reduced = pca.pca(np.absolute(strfs[i]), strfs[i].shape[1], n_components=n_components).flatten()
        tab_red.append(strf_reduced / np.max(strf_reduced))
    tab_red = np.transpose(np.asarray(tab_red))

    # optimization arguments
    optim_args = {
        'cost': 'correlation', 
        'init_sig_mean': 10.0,
        'init_sig_var': 1.0,
        'num_loops': 2000000,
    }
    # optimization
    correlations, sigmas = kernel_optim(tab_red, dissimil_mat, **optim_args)
    
    # log results in txt files and arguments in pickle file
    log_filename = tsp.lower()+'_'+time.strftime('%y%m%d-%H%M%S')
    np.savetxt('outs/'+log_filename+'_sigmas.txt', sigmas)
    np.savetxt('outs/'+log_filename+'_corrs.txt', correlations)
    optim_args.update({'pca_components': n_components})
    pickle.dump(optim_args, open('outs/'+log_filename+'_optim_args.pkl', 'wb'))

