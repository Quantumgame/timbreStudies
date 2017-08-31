import numpy as np
from optimization import kernel_optim


def get_representations(types=[]):
    representations = []
    if types == []:
        representations.extend(['STRF', 'FFT', 'STFT', 'CQT'])
        return ['STRF', 'FFT', 'STFT', 'CQT']
    if 'STRF' in types:
        representations.append('STRF')
    else:
        raise ValueError('Not implemented')
    return representations


def get_features(representation_name='STRF'):
    if representation_name == 'STRF':
        data = np.array([
            np.loadtxt('../tmpdata/data_reduced_sound%02i.txt' %
                       (sound_i + 1)).flatten() for sound_i in range(16)
        ]).T
        return data


def test_all():
    dissimilarities = np.loadtxt('../tmpdata/dissimilarity_matrix.txt')
    representations = get_representations(types=['STRF'])
    for rep_i, rep_name in enumerate(representations):
        features = get_features(rep_name)
        correlations = kernel_optim(features, dissimilarities, num_loops=1000)


if __name__ == "__main__":
    test_all()