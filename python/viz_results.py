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
    for lri, lr in enumerate(sorted(correlations.keys())):
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
        'iverson1993whole',
        'lakatoscomb',
        'lakatosharm',
        'lakatosperc',
        'mcadams1995',
        'patil2012_a3',
        'patil2012_dx4',
        'patil2012_gd4'
    ]
    representations = [
        'auditory_spectrum',
        'fourier_spectrum',
        # 'auditory_strf',
        # 'fourier_strf',
    ]
    files_ = {}
    dirs_ = {}
    for i, tsp in enumerate(sorted(timbrespace_db)):
        files_[tsp] = {}
        dirs_[tsp] = {}
        for r_i, repres in enumerate(representations):
            folder = 'outs_all_expopt/' + tsp.lower()
            files_[tsp][repres] = []
            dirs_[tsp][repres] = []
            for root, dirs, files in os.walk(folder):
                for f in files:
                    if repres in root.split('/')[-1] and f.split('=')[0] == 'optim_process_l':
                        files_[tsp][repres].append(f)
                        dirs_[tsp][repres] = root

    for i, tsp in enumerate(sorted(timbrespace_db)):
        plt.figure()
        for r_i, repres in enumerate(representations):
            # folder = 'outs_all/' + tsp.lower()
            # for root, dirs, files in os.walk(folder):
            #     test_dirs = []
            #     for f in files:
            #         if repres in root.split('/')[-1] and f.split('=')[0] == 'optim_process_l':
            #             # print('file', root, f)
            #             test_dirs.append(f)
            if len(files_[tsp][repres]):
                root = dirs_[tsp][repres]
                # print(repres, test_dirs)
                test_dirs = sorted(files_[tsp][repres])
                # print(test_dirs)
                file = test_dirs[-1]
                print(repres, os.path.join(root,file))
                optim_process = pickle.load(open(os.path.join(root,file), 'rb'))
                # corrs[tsp].append(optim_process['correlations'][-1])
                corrs = optim_process['correlations']
                # plt.plot(corrs[tsp], '-o')
                plt.subplot(1, 2, r_i+1)
                plt.plot(corrs, '-')
                plt.ylim([-1.0, 0.0])
        plt.show()


def compare_opt():
    timbrespace_db = [
        'grey1977',
        'grey1978',
        'iverson1993onset',
        'iverson1993whole',
        'lakatoscomb',
        'lakatosharm',
        'lakatosperc',
        'mcadams1995',
        'patil2012_a3',
        'patil2012_dx4',
        'patil2012_gd4'
    ]
    representations = [
        'auditory_spectrum',
        'fourier_spectrum',
        # 'auditory_strf',
        # 'fourier_strf',
    ]
    files_logs = {}
    dirs_logs = {}
    for i, tsp in enumerate(sorted(timbrespace_db)):
        files_logs[tsp] = {}
        dirs_logs[tsp] = {}
        for r_i, repres in enumerate(representations):
            folder = 'outs_all/' + tsp.lower()
            files_logs[tsp][repres] = {}
            dirs_logs[tsp][repres] = {}
            for root, dirs, files in os.walk(folder):
                for f in files:
                    if repres in root.split('/')[-1]:
                        oc = pickle.load(open(os.path.join(root,'optim_config.pkl'), 'rb'))
                        key = oc['args']['loss']+'-lr{}'.format(oc['args']['learning_rate'])
                        if key not in files_logs[tsp][repres].keys():
                            files_logs[tsp][repres][key] = []
                            dirs_logs[tsp][repres][key] = []
                        if f.split('=')[0] == 'optim_process_l':
                            # print(tsp, repres, key, f)
                            files_logs[tsp][repres][key].append(int(f.split('=')[1].split('.')[0]))
                            dirs_logs[tsp][repres][key] = root
                            # print(files_logs[tsp][repres][key])
    # print(files_logs['grey1977']['auditory_spectrum'])
    for i, tsp in enumerate(sorted(timbrespace_db)):
        plt.figure()
        plt.suptitle(tsp)
        for r_i, repres in enumerate(representations):
            if len(files_logs[tsp][repres]):
                plt.subplot(1, 2, r_i+1)
                for l in sorted(dirs_logs[tsp][repres].keys()):
                    root = dirs_logs[tsp][repres][l]
                    test_dirs = sorted(files_logs[tsp][repres][l])
                    # print(tsp, repres, test_dirs)
                    file = 'optim_process_l={}.pkl'.format(test_dirs[-1])
                    # print(test_dirs)
                    optim_process = pickle.load(open(os.path.join(root, file), 'rb'))
                    corrs = optim_process['correlations']
                    plt.plot(corrs, '-', label=l)
                plt.ylim([-1.0, 0.0])
                plt.legend()

        plt.show()





if __name__=="__main__":
    # viz_one_result()
    # viz_ntests()
    # res()
    compare_opt()

