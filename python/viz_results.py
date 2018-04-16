import numpy as np
import pickle
import os
import matplotlib.pylab as plt


def viz_one_result():
    folder = 'outs/_olds/grey1977'
    repres = 'auditory_spectrum'
    loop = 5000

    optim = pickle.load(open(os.path.join(folder,repres,'optim_process_l={}.pkl'.format(loop)), 'rb'))

    plt.plot(optim['correlations'])
    plt.show()


def viz_ntests():
    folder = 'outs/lakatoscomb'
    repres = 'auditory_spectrum'
    test_dirs = []
    for root, dirs, files in os.walk(folder):
        for d in dirs:
            if repres in d:
                test_dirs.append(os.path.join(root,d))
    print(test_dirs)
    loop = 40000
    correlations = {}
    # correlations['0.1'] = []
    for d in sorted(test_dirs):
        optim_config = pickle.load(open(os.path.join(d,'optim_config.pkl'.format(loop)), 'rb'))
        optim_process = pickle.load(open(os.path.join(d,'optim_process_l={}.pkl'.format(loop)), 'rb'))
        if ('learning_rate' in optim_config['args'].keys()):
            if str(optim_config['args']['learning_rate']) not in correlations.keys():
                correlations[str(optim_config['args']['learning_rate'])] = []
            correlations[str(optim_config['args']['learning_rate'])].append(optim_process['correlations'])
    n_lr = len(correlations.keys())
    for lri, lr in enumerate(correlations.keys()):
        plt.subplot(1,n_lr,lri+1)

        correlations_i = np.transpose(np.array(correlations[lr]))
        print(correlations_i.shape)
        # print(correlations_i.shape)
        means =  np.mean(correlations_i, axis=1)
        stds = np.std(correlations_i, axis=1)
        plt.title('{}, {:.5f}'.format(lr, means[-1]))
        # plt.plot(means)
        plt.plot(means, '-k')
        plt.plot(means+stds, '-r', linewidth=1)
        # print(means,stds)
        plt.plot(means-stds, '-r', linewidth=1)
        plt.ylim([-1.0, 0.0])
    plt.show()


def res():
    timbrespace_db = [
        'grey1977',
        'grey1978',
        'iverson1993onset',
    ]
    representations = [
        'auditory_spectrum',
        'fourier_spectrum',
        'auditory_strf',
        'fourier_strf',
    ]
    corrs = {}
    for i, tsp in enumerate(sorted(timbrespace_db)):
        corrs[tsp] = []
        for repres in representations:
            folder = 'outs_all/' + tsp.lower()
            for root, dirs, files in os.walk(folder):
                test_dirs = []
                for f in files:
                    if repres in root.split('/')[-1] and f.split('=')[0] == 'optim_process_l':
                        # print('file', root, f)
                        test_dirs.append(f)
                if len(test_dirs):
                    test_dirs = sorted(test_dirs)
                    # print(test_dirs)
                    file = test_dirs[-1]
                    # print(os.path.join(root,file))
                    optim_process = pickle.load(open(os.path.join(root,file), 'rb'))
                    corrs[tsp].append(optim_process['correlations'][-1])
        plt.plot(corrs[tsp], '-o')
    plt.show()




if __name__=="__main__":
    # viz_one_result()
    # viz_ntests()
    res()