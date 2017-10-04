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
import time


folder_list = ['Iverson93']

for folder in folder_list:

    # Compute or load STRFS
    if (not os.path.isfile('strfs.pkl')):
        strf_params = utils.get_strf_params()
        strfs = []
        for root, dirs, files in os.walk('../ext/'+folder):
            for name in files:
                if (name.split('.')[-1] == 'aiff'):
                    print('analysing', os.path.join(root, name))
                    strfs.append(STRF(os.path.join(root, name), **strf_params))
        pickle.dump(strfs, open('strfs.pkl', 'wb'))
    else:
        strfs = pickle.load(open('strfs.pkl', 'rb'))


    # Load dissimilarity matrix
    dissimil_mat = utils.get_dissimalrity_matrix('../ext/'+folder)


    # PCA
    counter = 1
    for iterrr in range (1,2):
        for n_components in range(1,2):

            tab_red = []
            # n_components = 1
            for i in range(len(strfs)):
                print('PCA on sound %02i' % (i + 1))
                strf_reduced = pca.pca(np.absolute(strfs[i]), strfs[i].shape[1], n_components=n_components).flatten()
                tab_red.append(strf_reduced / np.max(strf_reduced))
            tab_red = np.transpose(np.asarray(tab_red))
            

            for m in [5.0, 10.0, 50.0, 100.0, 500.0]:
                for s in [0.5, 1.0, 10., 50.0]:
                    print('Test',counter,'out of',20,':',m,s,n_components)
                    counter += 1

                    # pptimization argumenrs
                    optim_args = {
                        'cost': 'correlation', 
                        'init_sig_mean': m,
                        'init_sig_var': s,
                        'num_loops': 30000,
                    }

                    # optimization
                    correlations, sigmas = kernel_optim(tab_red, dissimil_mat, **optim_args)
                    
                    # log results in txt files and arguments in pickle file
                    log_filename = folder.lower()+'_'+time.strftime('%y%m%d-%H%M%S')
                    np.savetxt(log_filename+'_sigmas.txt', sigmas)
                    np.savetxt(log_filename+'_corrs.txt', correlations)
                    optim_args.update({'n_components': n_components})
                    pickle.dump(optim_args, open(log_filename+'_optim_args.pkl', 'wb'))


