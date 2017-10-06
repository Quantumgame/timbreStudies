# Copyright (c) Baptiste Caramiaux, Etienne Thoret
# All rights reserved

import numpy as np


def adhoc_pca(data, num_comp):
    n, p = data.shape[0], data.shape[1]
    # subtract off the mean for each dimension
    mn = np.mean(data, axis=0)
    data = np.subtract(data, mn)
    # construct the matrix Y
    Y = data / np.sqrt(n - 1)
    # SVD does it all
    u, S, pc = np.linalg.svd(Y, full_matrices=False)
    # project the original data
    points = np.transpose(np.dot(pc, np.transpose(data)))
    # calculate the variances
    # variances = np.multiply(S, S)
    # find minimum nb of components needed to have var > threshold
    # cum_explained = np.cumsum(variances / np.sum(variances))
    # idx = np.where(cum_explained > threshold)[0]
    # return projected data
    # return points[:, :idx[0] + 1]
    return points[:, :num_comp]


def pca(tensor_strf, nb_freq, n_components=1):
    num_comp = n_components
    tensor_strf_avg = np.mean(tensor_strf, axis=0)
    strf_pca = np.zeros((nb_freq, num_comp*tensor_strf.shape[2]))
    for freq_i in range(nb_freq):
        strf_pca[freq_i, :] = np.transpose(
            adhoc_pca(tensor_strf_avg[freq_i, :, :], num_comp).flatten())
    return strf_pca