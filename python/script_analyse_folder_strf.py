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


# Compute or load STRFS
if (not os.path.isfile('strfs.pkl')):
    strf_params = utils.get_strf_params()
    strfs = []
    for root, dirs, files in os.walk('../ext/sounds'):
        for name in files:
            if (name.split('.')[-1] == 'aiff'):
                print('analysing', os.path.join(root, name))
                strfs.append(STRF(os.path.join(root, name), **strf_params))
    pickle.dump(strfs, open('strfs.pkl', 'wb'))
else:
    strfs = pickle.load(open('strfs.pkl', 'rb'))


# Load dissimilarity matrix
dissimil_mat = utils.get_dissimalrity_matrix('../ext/data/')


# PCA
tab_red = []
for i in range(len(strfs)):
    print('PCA on sound %02i' % (i + 1))
    strf_reduced = pca.pca(np.absolute(strfs[i]), strfs[i].shape[1]).flatten()
    tab_red.append(strf_reduced / np.max(strf_reduced))
tab_red = np.transpose(np.asarray(tab_red))


# Optimization
correlations = kernel_optim(tab_red, dissimil_mat, num_loops=10000)
plt.plot(correlations)
plt.show()
