import numpy as np
import pickle
import math
import os
import matplotlib.pylab as plt
from lib import load

# def viz_single_corr(folder = 'outs/validtests_06-21-2018/grey1978/fourier_spectrum-180621@005541'):
#     optim = pickle.load(open(os.path.join(folder, 'optim_process_l=1000.pkl'), 'rb'))
#     plt.plot(optim['correlations'])
#     plt.show()


# def viz_single_sigmas(folder = 'outs/grey1977/auditory_strf-180622@142544'):
#     optim = pickle.load(open(os.path.join(folder, 'optim_process_l=400.pkl'), 'rb'))
#     plt.plot(optim['sigmas'])
#     plt.show()


timbre_spaces= list(sorted(load.database().keys()))
audio_representations = sorted([
    'auditory_spectrum', 'fourier_spectrum', 'auditory_strf',
    'fourier_strf', 'auditory_spectrogram', 'fourier_spectrogram',
    'auditory_mps', 'fourier_mps'
])


def correlation_between_timbrespaces(res_path='outs/validtests_06-21-2018'):
    timbrespaces = []
    representations = []
    loops = {}
    sigmas = {}
    for root, dirs, files in os.walk(res_path):
        for f in files:
            if 'optim_process' in f:
                if root.split('/')[-2] not in loops.keys():
                    loops[root.split('/')[-2]] = {}
                    sigmas[root.split('/')[-2]] = {}
                loopid = int(f.split('=')[-1].split('.')[0])
                sigmas_ = pickle.load(open(os.path.join(root, f), 'rb'))['sigmas']
                if root.split('/')[-1].split('-')[0] not in loops[root.split('/')[-2]].keys():
                    loops[root.split('/')[-2]][root.split('/')[-1].split('-')[0]] = []
                    sigmas[root.split('/')[-2]][root.split('/')[-1].split('-')[0]] = []
                loops[root.split('/')[-2]][root.split('/')[-1].split('-')[0]].append(loopid)
                sigmas[root.split('/')[-2]][root.split('/')[-1].split('-')[0]].append(sigmas_)

    keys = list(sorted(loops.keys()))
    key0 = keys[0]
    corrs = []
    for rs in sorted(loops[key0].keys()):
        corrs_ = []
        for ts_i in range(0, len(keys)-1):
            for ts_j in range(1, len(keys)):
                loop_max_i = np.argmax(loops[keys[ts_i]][rs])
                sigmas_i = sigmas[keys[ts_i]][rs][loop_max_i]
                loop_max_j = np.argmax(loops[keys[ts_j]][rs])
                sigmas_j = sigmas[keys[ts_j]][rs][loop_max_j]
                corrc = np.corrcoef(sigmas_i, sigmas_j)[0,1]
                corrs_.append(corrc)
        corrs.append(np.mean(corrs_))
    x = np.arange(len(corrs))
    plt.figure(figsize=(12,8))
    plt.plot(corrs, '-ok')
    plt.xticks(x, [s.replace('_', '\n') for s in sorted(loops[key0].keys())]) #, rotation='vertical')
    plt.ylabel('correlation')
    plt.savefig(
        'correlation_cross_ts.pdf',
        bbox_inches='tight')
    plt.show()


def test_sigma_transfer(res_path='outs/ts_crossval'):
    corrs = []
    labels = []
    for rs_i, rs in enumerate(audio_representations):
        print(rs)
        corrs_rs = []
        for ts_i in range(0, len(timbre_spaces)):
            print(' ', timbre_spaces[ts_i])
            input_data = np.loadtxt(os.path.join(res_path, '{}_{}_input_data.txt'.format(timbre_spaces[ts_i], rs)))
            target_data = np.loadtxt(os.path.join(res_path, '{}_{}_target_data.txt'.format(timbre_spaces[ts_i], rs)))
            for ts_j in range(0, len(timbre_spaces)):
                if ts_i != ts_j:
                    sigmas = np.loadtxt(os.path.join(res_path, '{}_{}_sigmas.txt'.format(timbre_spaces[ts_j], rs)))
                    ndims, ninstrus = input_data.shape[0], input_data.shape[1]
                    no_samples = ninstrus * (ninstrus - 1) / 2
                    idx_triu = np.triu_indices(target_data.shape[0], k=1)
                    target_v = target_data[idx_triu]
                    mean_target = np.mean(target_v)
                    std_target = np.std(target_v)
                    kernel = np.zeros((ninstrus, ninstrus))
                    for i in range(ninstrus):
                            for j in range(i + 1, ninstrus):
                                kernel[i, j] = -np.sum(
                                    np.power(
                                        np.divide(input_data[:, i] - input_data[:, j],
                                                  (sigmas + np.finfo(float).eps)), 2))
                    kernel_v = kernel[idx_triu]
                    mean_kernel = np.mean(kernel_v)
                    std_kernel = np.std(kernel_v)
                    Jn = np.sum(np.multiply(kernel_v - mean_kernel, target_v - mean_target))
                    Jd = no_samples * std_target * std_kernel
                    corrs_rs.append(Jn/Jd)
        corrs.append(corrs_rs)
        labels.append(rs.replace('_', '\n'))
    x = np.arange(len(corrs))
    idx = sorted(range(len(labels)), key=lambda k: labels[k])
    sorted_corrs = [np.mean(corrs[k]) for k in idx]
    sorted_corrs_std = [np.std(corrs[k]) for k in idx]
    sorted_labels = [labels[k] for k in idx]
    plt.figure(figsize=(12,8))
    plt.plot(sorted_corrs, '-ok')
    for xi in x:
        plt.plot([xi, xi], [sorted_corrs[xi], sorted_corrs[xi] + sorted_corrs_std[xi]], '--k', linewidth=1)
        plt.plot([xi, xi], [sorted_corrs[xi], sorted_corrs[xi] - sorted_corrs_std[xi]], '--k', linewidth=1)
    plt.xticks(x, sorted_labels) #, rotation='vertical')
    plt.ylabel('correlation')
    plt.savefig(
        'test_sigma_transfer.pdf',
        bbox_inches='tight')
    plt.show()


def correlation_sigmas_crosstrained_vs_withintrained(res_path='outs/ts_crossval'):
    corrs = []
    labels = []
    for rs_i, rs in enumerate(audio_representations):
        crossts_sigma = np.loadtxt(os.path.join(res_path, 'crossts_{}_sigmas.txt'.format(rs)))
        corrs_ = []
        for ts_i in range(0, len(timbre_spaces)-1):
            sigmas_i = np.loadtxt(os.path.join(res_path, '{}_{}_sigmas.txt'.format(timbre_spaces[ts_i], rs)))
            corrs_.append(np.corrcoef(crossts_sigma, sigmas_i)[0,1])
        corrs.append(np.mean(corrs_))
        labels.append(rs.replace('_', '\n'))
    x = np.arange(len(corrs))
    plt.figure(figsize=(12,8))
    idx = sorted(range(len(labels)), key=lambda k: labels[k])
    sorted_corrs = [corrs[k] for k in idx]
    sorted_labels = [labels[k] for k in idx]
    plt.plot(sorted_corrs, '-ok')
    plt.xticks(x, sorted_labels) #, rotation='vertical')
    plt.ylabel('correlation')
    plt.savefig(
        'correlation_sigmas_crosstrained_vs_withintrained.pdf',
        bbox_inches='tight')
    plt.show()


def correlation_cross_timbrespaces(res_path='outs/ts_crossval'):
    labels = []
    correlations = []
    for rs_i, rs in enumerate(audio_representations):
        correlations.append(np.loadtxt(os.path.join(res_path, 'crossts_{}_correlations.txt'.format(rs)))[-1])
        labels.append(rs.replace('_', '\n'))

    corrs = []
    for rs_i, rs in enumerate(audio_representations):
        corrs_ = []
        for ts_i in range(0, len(timbre_spaces)-1):
            for ts_j in range(1, len(timbre_spaces)):
                sigmas_i = np.loadtxt(os.path.join(res_path, '{}_{}_sigmas.txt'.format(timbre_spaces[ts_i], rs)))
                sigmas_j = np.loadtxt(os.path.join(res_path, '{}_{}_sigmas.txt'.format(timbre_spaces[ts_j], rs)))
                corrs_.append(np.corrcoef(sigmas_i, sigmas_j)[0,1])
        corrs.append(np.mean(corrs_))

    sorted_corrs_optim = correlations
    sorted_corrs_comp = corrs
    sorted_labels = labels
    # idx = sorted(range(len(labels)), key=lambda k: labels[k])
    # sorted_corrs_optim = [correlations[k] for k in idx]
    # sorted_corrs_comp = [corrs[k] for k in idx]
    # sorted_labels = [labels[k] for k in idx]
    x = np.arange(len(correlations))
    plt.figure(figsize=(12,8))
    # plt.plot(sorted_corrs_optim, '-ok')
    plt.plot(sorted_corrs_comp, '-ok')
    plt.xticks(x, sorted_labels) #, rotation='vertical')
    plt.ylabel('correlation')
    plt.savefig(
        'test_correlation_between_sigmas_crossts.pdf',
        bbox_inches='tight')
    plt.show()


def correlation_sanity_check():
    # correlation_greys()
    timbrespace = 'grey1978'
    # folder = 'auditory_spectrogram-180503@204215'
    folder = 'auditory_spectrogram-180503@213954'
    optim_res = pickle.load(
        open('tests/durations/' + timbrespace + '/' + folder +
             '/optim_process_l=1000.pkl', 'rb'))
    sigmas = optim_res['sigmas']
    plt.plot(sigmas)
    plt.title('sigmas')
    plt.show()

    dataset = pickle.load(
        open('tests/durations/' + timbrespace + '/' + folder + '/dataset.pkl',
             'rb'))
    input_data = dataset['data_proj']
    input_data = input_data / np.mean(np.max(np.abs(input_data), axis=0))
    plt.plot(input_data)
    plt.title('input_data (sizes={}, orig_size={})'.format(
        input_data.shape, dataset['data_repres'][0].shape))
    plt.show()

    # plt.plot(np.divide(input_data, sigmas.reshape(-1,1)))
    # plt.title('weighted data')
    # plt.show()

    # plt.imshow(np.divide(input_data[:,0], sigmas).reshape(1024,24), interpolation='nearest', aspect='auto')
    # plt.title('weighted data')
    # plt.show()

    ndims, ninstrus = input_data.shape[0], input_data.shape[1]
    no_samples = ninstrus * (ninstrus - 1) / 2

    target_data = dataset['dissimilarities']
    idx_triu = np.triu_indices(target_data.shape[0], k=1)
    target_v = target_data[idx_triu]
    mean_target = np.mean(target_v)
    std_target = np.std(target_v)

    kernel = np.zeros((ninstrus, ninstrus))
    for i in range(ninstrus):
        for j in range(i + 1, ninstrus):
            dist = np.power(
                np.divide(input_data[:, i] - input_data[:, j], sigmas), 2)
            kernel[i, j] = np.exp(-np.sum(dist))

    kernel_v = kernel[idx_triu]
    mean_kernel = np.mean(kernel_v)
    std_kernel = np.std(kernel_v)
    Jn = np.sum(np.multiply(kernel_v - mean_kernel, target_v - mean_target))
    Jd = (no_samples - 1) * std_target * std_kernel

    plt.suptitle('{} - {}'.format(Jn / Jd, optim_res['Jn'] / optim_res['Jd']))
    plt.subplot(1, 2, 1)
    plt.imshow(-1.0 * np.log(kernel))
    plt.subplot(1, 2, 2)
    plt.imshow(target_data)
    plt.show()

    # plt.plot(optim_res['correlations'])
    # plt.show()


# def compare_opt_lgbfs():
#     timbrespace_db = [
#         'grey1977', 'grey1978', 'iverson1993onset', 'iverson1993whole',
#         'lakatoscomb', 'lakatosharm', 'lakatosperc', 'mcadams1995',
#         'patil2012_a3', 'patil2012_dx4', 'patil2012_gd4'
#     ]
#     representations = [
#         'auditory_spectrum',
#         'fourier_spectrum',
#         # 'auditory_strf',
#         # 'fourier_strf',
#         # 'auditory_mps',
#         # 'fourier_mps',
#     ]
#     files_logs = {}
#     dirs_logs = {}
#     for i, tsp in enumerate(sorted(timbrespace_db)):
#         files_logs[tsp] = {}
#         dirs_logs[tsp] = {}
#         for r_i, repres in enumerate(representations):
#             folder = 'outs_lbfgs/' + tsp.lower()
#             files_logs[tsp][repres] = {}
#             dirs_logs[tsp][repres] = {}
#             for root, dirs, files in os.walk(folder):
#                 for f in files:
#                     if repres in root.split('/')[-1]:
#                         oc = pickle.load(
#                             open(os.path.join(root, 'optim_config.pkl'), 'rb'))
#                         # print(tsp, repres, root)
#                         key = 'L-BFGS-B'  #oc['args']['loss']+'-lr{}'.format(oc['args']['learning_rate'])
#                         if key not in files_logs[tsp][repres].keys():
#                             files_logs[tsp][repres][key] = []
#                             dirs_logs[tsp][repres][key] = []
#                         if f.split('=')[0] == 'optim_process_l':
#                             # print(tsp, repres, key, f)
#                             files_logs[tsp][repres][key].append(
#                                 int(f.split('=')[1].split('.')[0]))
#                             dirs_logs[tsp][repres][key] = root
#                             # print(files_logs[tsp][repres][key])
#                         # print(files_logs['grey1977']['auditory_spectrum'])
#     for i, tsp in enumerate(sorted(timbrespace_db)):
#         plt.figure(figsize=(12, 6))
#         plt.suptitle(tsp)
#         for r_i, repres in enumerate(representations):
#             if len(files_logs[tsp][repres]):
#                 plt.subplot(math.ceil(len(representations) / 2), 2, r_i + 1)
#                 for l in sorted(dirs_logs[tsp][repres].keys()):
#                     root = dirs_logs[tsp][repres][l]
#                     test_dirs = sorted(files_logs[tsp][repres][l])
#                     # print(tsp, repres, test_dirs)
#                     file = 'optim_process_l={}.pkl'.format(test_dirs[-1])
#                     # print(tsp, repres, file)
#                     # print(test_dirs)
#                     optim_process = pickle.load(
#                         open(os.path.join(root, file), 'rb'))
#                     corrs = optim_process['correlations']
#                     plt.plot(corrs, '-', label=l)
#                     plt.title(repres)
#                 plt.ylim([-1.0, 0.0])
#                 # plt.xlim([0, 50000])
#                 plt.legend()

#         plt.show()
# 





def final_correlations(path='./'):
    # files_logs = {}
    # dirs_logs = {}
    # for i, tsp in enumerate(sorted(timbre_spaces)):
    #     files_logs[tsp] = {}
    #     dirs_logs[tsp] = {}
    #     for r_i, repres in enumerate(audio_representations):
    #         folder = os.path.join(path, tsp.lower())
    #         files_logs[tsp][repres] = []
    #         dirs_logs[tsp][repres] = []
    #         for root, dirs, files in os.walk(folder):
    #             for f in files:
    #                 if repres in root.split('/')[-1]:
    #                     oc = pickle.load(
    #                         open(os.path.join(root, 'optim_config.pkl'), 'rb'))
    #                     if f.split('=')[0] == 'optim_process_l':
    #                         files_logs[tsp][repres].append(int(f.split('=')[1].split('.')[0]))
    #                         dirs_logs[tsp][repres] = root
    allcorrs = []
    for i, ts in enumerate(timbre_spaces):
        corrs_tsp = []
        labels = []
        for r_i, rs in enumerate(audio_representations):
            corrs = np.loadtxt(os.path.join(path,'{}_{}_correlations.txt'.format(ts,rs)))[-1]
            # print(tsp, repres, file)
            # optim_process = pickle.load(
            #     open(os.path.join(root, file), 'rb'))
            # corrs = optim_process['Jn'] / optim_process['Jd']
            corrs_tsp.append(corrs)
            labels.append(rs.replace('_', '\n'))
        plt.plot(corrs_tsp, '-o', label=ts)
        allcorrs.append(corrs_tsp)
    lgd = plt.legend(bbox_to_anchor=(1.05, 1.0))
    plt.xticks(np.arange(len(audio_representations)), labels, rotation='vertical')
    plt.ylim([0.0, -1.1])
    plt.ylabel('Correlation')
    plt.savefig(
        'final_correlation_allspaces.pdf',
        bbox_extra_artists=(lgd, ),
        bbox_inches='tight')
    plt.show()

    allcorrs = np.array(allcorrs)
    mus = np.mean(allcorrs, axis=0)
    sts = np.std(allcorrs, axis=0)
    plt.plot(mus, 'ok')
    for k in range(len(mus)):
        plt.plot([k,k], [mus[k],mus[k]+sts[k]], '-k', linewidth=1)
        plt.plot([k,k], [mus[k],mus[k]-sts[k]], '-k', linewidth=1)
    plt.xticks(np.arange(len(audio_representations)), labels, rotation='vertical')
    plt.ylim([0.0, -1.1])
    plt.ylabel('Correlation')
    plt.savefig(
        'final_correlation_byrepres.pdf',
        bbox_inches='tight')
    plt.show()

    mus = np.mean(allcorrs, axis=1)
    sts = np.std(allcorrs, axis=1)
    plt.plot(mus, 'ok')
    for k in range(len(mus)):
        plt.plot([k,k], [mus[k],mus[k]+sts[k]], '-k', linewidth=1)
        plt.plot([k,k], [mus[k],mus[k]-sts[k]], '-k', linewidth=1)
    plt.xticks(np.arange(len(sorted(timbre_spaces))), sorted(timbre_spaces), rotation='vertical')
    plt.ylim([0.0, -1.1])
    plt.ylabel('Correlation')
    plt.savefig(
        'final_correlation_bytimbrespaces.pdf',
        bbox_inches='tight')
    plt.show()


def analyze_tests(path='tests', method='L-BFGS-B'):
    test_type = 'offset'
    path += '/'+test_type+'s'
    cases = []
    for root, dirs, files in os.walk(path):
        for d in dirs:
            if '=' in d:
                cases.append(d.split('=')[-1])
    # print(cases)
    timbrespaces = []
    representations = []
    for root, dirs, files in os.walk(path):
        for d in dirs:
            if '@' not in d and '=' not in d:
                timbrespaces.append(d)
            if '@' in d:
                representations.append(d.split('-')[0])
    # representations = [
    #     'auditory_spectrum',
    #     'fourier_spectrum',
    #     'auditory_spectrogram',
    #     'fourier_spectrogram',
    #     'auditory_mps',
    #     'fourier_mps',
    #     'auditory_strf',
    #     'fourier_strf',
    # ]
    # if (not os.path.isfile('files_dirs_for_results.pkl')):
    files_logs = {}
    dirs_logs = {}
    for i, tsp in enumerate(sorted(timbrespaces)):
        files_logs[tsp] = {}
        dirs_logs[tsp] = {}
        for r_i, repres in enumerate(sorted(representations)):
            files_logs[tsp][repres] = {}
            dirs_logs[tsp][repres] = {}
            for d_i, dur in enumerate(sorted(cases)):
                folder = os.path.join(path, test_type + '=' + dur, tsp.lower())
                files_logs[tsp][repres][dur] = []
                dirs_logs[tsp][repres][dur] = []
                for root, dirs, files in os.walk(folder):
                    for f in files:
                        if repres in root.split('/')[-1]:
                            oc = pickle.load(
                                open(
                                    os.path.join(root, 'optim_config.pkl'),
                                    'rb'))
                            if f.split('=')[0] == 'optim_process_l':
                                files_logs[tsp][repres][dur].append(
                                    int(f.split('=')[1].split('.')[0]))
                                dirs_logs[tsp][repres][dur] = root

    for i, tsp in enumerate(sorted(timbrespaces)):
        plt.figure()
        # plt.suptitle(tsp)
        for d_i, dur in enumerate(sorted(cases)):
            corrs_tsp = []
            labels = []
            for r_i, repres in enumerate(sorted(representations)):
                if len(files_logs[tsp][repres][dur]):
                    root = dirs_logs[tsp][repres][dur]
                    test_dirs = sorted(files_logs[tsp][repres][dur])
                    file = 'optim_process_l={}.pkl'.format(test_dirs[-1])
                    optim_process = pickle.load(
                        open(os.path.join(root, file), 'rb'))
                    corrs = optim_process['Jn'] / optim_process['Jd']
                    corrs_tsp.append(corrs)
                    labels.append(repres.replace('_', '\n'))
            plt.plot(corrs_tsp, '-o', label=dur)
        lgd = plt.legend(bbox_to_anchor=(1.2, 1.0))
        plt.xticks(
            np.arange(len(representations)), labels, rotation='vertical')
        plt.ylim([0.0, -1.05])
        plt.ylabel('Correlation')
        plt.savefig(
            'test_{}_{}.pdf'.format(test_type,tsp),
            bbox_extra_artists=(lgd, ),
            bbox_inches='tight')
        # plt.show()


def analyse_one():
    folder = '/Volumes/LaCie/outputs'
    tsp = 'iverson1993whole'
    rep = 'fourier_mps-180422@185712'
    loop = 500
    # data = pickle.load(open(os.path.join(folder, tsp, rep, 'dataset.pkl'), 'rb'))
    con = pickle.load(
        open(os.path.join(folder, tsp, rep, 'optim_config.pkl'), 'rb'))
    op = pickle.load(
        open(
            os.path.join(folder, tsp, rep, 'optim_process_l={}.pkl'.format(
                loop)), 'rb'))
    print(con['optim_options']['bounds'][0][1], op['correlations'][-1])
    plt.plot(op['sigmas'])
    plt.title('{}'.format(op['correlations'][-1]))
    plt.show()
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(data['data_proj'], interpolation='nearest', aspect='auto')
    plt.subplot(1, 2, 2)
    data_weighted = np.divide(data['data_proj'], op['sigmas'].reshape(-1, 1))
    plt.imshow(data_weighted, interpolation='nearest', aspect='auto')
    plt.show()


if __name__ == "__main__":
    
    # final_correlations(path='results_07-26-2018')
    # correlation_cross_timbrespaces(res_path='results_07-26-2018')
    # correlation_between_timbrespaces()
    # test_sigma_transfer(res_path='results_07-26-2018')
    correlation_sigmas_crosstrained_vs_withintrained(res_path='results_07-26-2018')
    
    
    # analyze_tests()
    # analyse_one()
    # correlation_sanity_check()
