import os
import pickle
import numpy as np
from lib import utils
from lib import auditory
from lib import fourier
import subprocess


def convert(data_path='./outs/validtests_06-21-2018', output_path='results_07-26-2018'):
    subprocess.call(['mkdir', '-p', output_path])
    timbrespace_db = []
    for root, dirs, files in os.walk(data_path):
        for name in dirs:
            if '@' not in name:
                the_timbrespace = {}
                the_timbrespace['name'] = name
                the_timbrespace['path'] = os.path.join(root, name)
                timbrespace_db.append(the_timbrespace)
                print(name)
    for ts in timbrespace_db:
        for root, dirs, files in os.walk(ts['path']):
            loops = []
            for file in files:
                if 'process' in file:
                    loops.append(int(file.split('=')[1].split('.')[0]))
            if len(loops):
                repres = root.split('/')[-1].split('-')[0]
                
                dataset = pickle.load(open(os.path.join(root, 'dataset.pkl'), 'rb'))
                process = pickle.load(open(os.path.join(root, 'optim_process_l={}.pkl'.format(np.max(loops))), 'rb'))
                # if len(dataset['data_repres'][0].shape)>1:
                #     lll = dataset['data_repres'][0].shape[0] * dataset['data_repres'][0].shape[1]
                # else:
                #     lll = dataset['data_repres'][0].shape[0]
                # if 'strf' not in root.split('/')[-1].split('-')[0]:
                #     print(ts['name'], '\t',  root.split('/')[-1].split('-')[0], '\t representation_size:{}'.format(dataset['data_repres'][0].shape), '\t input_data_size:{}'.format(dataset['data_proj'].shape))
                # print(dataset['data_repres'][0].shape[0] * dataset['data_repres'][0].shape[1])
                np.savetxt(os.path.join(output_path, '{}_{}_input_data.txt'.format(ts['name'], repres)), dataset['data_proj'])
                np.savetxt(os.path.join(output_path, '{}_{}_target_data.txt'.format(ts['name'], repres)), dataset['dissimilarities'])
                np.savetxt(os.path.join(output_path, '{}_{}_sigmas.txt'.format(ts['name'], repres)), process['sigmas'])
                np.savetxt(os.path.join(output_path, '{}_{}_correlations.txt'.format(ts['name'], repres)), process['correlations'])
                pickle.dump(dataset['mapping'], open(os.path.join(output_path, '{}_{}_mappings.pkl'.format(ts['name'], repres)), 'wb'))


def manage_crossts_results(data_path='./outs/ts_crossval', output_path='results_07-26-2018'):
    subprocess.call(['mkdir', '-p', output_path])
    repres_db=[]
    for root, dirs, files in os.walk(data_path):
        for name in dirs:
            if '@' in name:
                the_repres = {}
                the_repres['name'] = name.split('-')[0]
                the_repres['path'] = os.path.join(root, name)
                repres_db.append(the_repres)
    for rs in repres_db:
        for root, dirs, files in os.walk(rs['path']):
            loops = []
            for file in files:
                if 'process' in file:
                    loops.append(int(file.split('=')[1].split('.')[0]))
            if len(loops):
                repres = rs['name']
                process = pickle.load(open(os.path.join(root, 'optim_process_l={}.pkl'.format(np.max(loops))), 'rb'))
                np.savetxt(os.path.join(output_path, 'crossts_{}_sigmas.txt'.format(rs['name'])), process['sigmas'])
                # np.savetxt(os.path.join(output_path, 'crossts_{}_correlations.txt'.format(rs['name'])), [process['Jn']/process['Jd']])
                np.savetxt(os.path.join(output_path, 'crossts_{}_correlations.txt'.format(rs['name'])), process['correlations'])


if __name__ == '__main__':
    # convert()
    manage_crossts_results()