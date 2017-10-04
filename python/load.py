# Copyright (c) Baptiste Caramiaux, Etienne Thoret
# All rights reserved

import os
import pickle
import numpy as np
import utils
from features import STRF


def load_timbrespace_database(root_path='../dataSoundsDissim/'):
    timbrespace_db = {}
    for root, dirs, files in os.walk(os.path.join(root_path, 'sounds')):
        for name in dirs:
            timbrespace_db[name] = {}
            timbrespace_db[name]['path'] = os.path.join(root, name)
    return timbrespace_db


def timbrespace_strf(timbrespace, timbrespace_db):
    valid_timbrespace_name = timbrespace in timbrespace_db.keys()
    if not valid_timbrespace_name:
        raise ValueError('{} is a wrong timbre space name'.format(timbrespace))
    if (not os.path.isfile(
            os.path.join('processed_data', timbrespace + '_strfs.pkl'))):
        strf_params = utils.get_strf_params()
        strfs = []
        for root, dirs, files in os.walk(el['path']):
            for name in files:
                if (name.split('.')[-1] == 'aiff'):
                    print('analysing',
                          os.path.join(timbrespace_db[timbrespace]['path'],
                                       name))
                    strfs.append(STRF(os.path.join(root, name), **strf_params))
        pickle.dump(strfs,
                    open(
                        os.path.join('processed_data',
                                     timbrespace + '_strfs.pkl'), 'wb'))
    else:
        strfs = pickle.load(
            open(
                os.path.join('processed_data', timbrespace + '_strfs.pkl'),
                'rb'))
    return strfs


def timbrespace_dismatrix(timbrespace, timbrespace_db):
    valid_timbrespace_name = timbrespace in timbrespace_db.keys()
    if not valid_timbrespace_name:
        raise ValueError('{} is a wrong timbre space name'.format(timbrespace))
    root_path = os.path.join(*timbrespace_db[timbrespace]['path'].split('/')[:-2])
    if os.path.isfile(
            os.path.join(root_path, 'data', timbrespace +
                         '_dissimilarity_matrix.txt')):
        return np.loadtxt(
            os.path.join(root_path, 'data', timbrespace +
                         '_dissimilarity_matrix.txt'))


def load_dismatrices(root_path='../dataSoundsDissim/'):
    timbrespace_names = load_timbrespace_names(root_path)
    dismatrices = []
    for i, el in enumerate(timbrespace_names):
        if os.path.isfile(
                os.path.join(root_path, 'data', el['name'] +
                             '_dissimilarity_matrix.txt')):
            dismatrices.append({
                'name': el['name'], \
                'matrix': np.loadtxt(
                            os.path.join(root_path, 'data', el['name'] +
                                 '_dissimilarity_matrix.txt'))
            })
    print(len(dismatrices), dismatrices[0]['name'], dismatrices[0]['matrix'])


if __name__ == "__main__":
    load_dismatrices()
    # print(load_timbrespace_names())