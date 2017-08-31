import numpy as np
import matplotlib.pylab as plt


def kernel_optim(input_data, target_data, cost='correlation', num_loops=20000):

    ndims, ninstrus = input_data.shape[0], input_data.shape[1]
    no_samples = ninstrus * (ninstrus - 1) / 2
    grad_corrfunc = np.zeros((ndims, 1))
    sigmas = 10.0 + 0.5 * np.random.randn(ndims, 1)

    correlations = np.zeros((num_loops, 1))

    idx_triu = np.triu_indices(target_data.shape[0], k=1)
    target_v = target_data[idx_triu]
    mean_target = np.mean(target_v)
    std_target = np.std(target_v)

    kernel = np.zeros((ninstrus, ninstrus))
    dkernel = np.zeros((ninstrus, ninstrus, ndims))

    for loop in range(num_loops):  # 0 to nump_loops-1
        sigmas = sigmas - grad_corrfunc * sigmas
        for i in range(ninstrus):
            for j in range(i + 1, ninstrus):
                kernel[i, j] = np.exp(-np.sum(
                    np.power(
                        np.divide(input_data[:, i] - input_data[:, j],
                                  sigmas[:, 0]), 2)))
                dkernel[i, j, :] = 2 * kernel[i, j] * np.power(
                    (input_data[:, i] - input_data[:, j]), 2) / np.power(
                        sigmas[:, 0], 3)
        kernel_v = kernel[idx_triu]
        mean_kernel = np.mean(kernel_v)
        std_kernel = np.std(kernel_v)

        Jn = np.sum(
            np.multiply(kernel_v - mean_kernel, target_v - mean_target))
        Jd = (no_samples - 1) * std_target * std_kernel
        correlations[loop] = Jn / Jd

        for k in range(ndims):
            dkernel_k_v = dkernel[:, :, k][idx_triu]
            dJn = np.sum(dkernel_k_v * (target_v - mean_target))
            dJd = (no_samples - 1) / no_samples * \
                        std_target / std_kernel * \
                        np.sum(dkernel_k_v * (kernel_v - mean_kernel))
            grad_corrfunc[k] = (Jd * dJn - Jn * dJd) / (Jd**2)
        # verbose
        if ((loop + 1) % 100 == 0):
            print('loop=%d | grad=%.6f | J=%.6f' %
                  (loop + 1, np.linalg.norm(grad_corrfunc, 2),
                   correlations[loop]))
    return correlations

if __name__ == "__main__":
    data_dir = '../tmpdata/'
    data = []
    for sound_i in range(16):
        data_mat = np.loadtxt(data_dir + 'data_reduced_sound%02i.txt' %
                              (sound_i + 1))
        data.append(data_mat.flatten())
    input_data = np.array(data).T
    target_data = np.loadtxt(data_dir + 'dissimilarity_matrix.txt')
    correlations = kernel_optim(input_data, target_data, num_loops=1000)
    plt.plot(correlations)
    plt.show()