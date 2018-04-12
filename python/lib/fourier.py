'''
Copyright (c) Baptiste Caramiaux, Etienne Thoret
All rights reserved

'''
import numpy as np
import math
from scipy import signal
from lib import utils
from lib import features  #import spectrum2scaletime, scaletime2scalerate, scalerate2cortical, waveform2auditoryspectrogram


def load_params():
    params = {}
    params['windowSize'] = 743
    params['frameStep'] = 185
    params['durationCut'] = 0.3
    params['durationRCosDecay'] = 0.05
    params['newFs'] = 16000
    return params

def gaussianWdw2d(mu_x, sigma_x, mu_y, sigma_y, x, y):
    # %window = exp(-(x - mu_x).^2 / 2 / sigma_x / sigma_x) .* exp(-(y - mu_y).^2 / 2 / sigma_y / sigma_y) ;
    m1 = np.exp(-np.power((x - mu_x),2) / 2 / sigma_x / sigma_x).reshape(-1,1)
    m2 = np.exp(-np.power(y - mu_y,2) / 2 / sigma_y / sigma_y).reshape(1,-1)
    window = np.dot(m1, m2)
    # print(window.shape)
    return window


def spectrum(wavtemp, fs):
    # load analysis parameters
    params = load_params()
    new_fs = params['newFs']
    windowSize = params['windowSize']
    frameStep = params['frameStep']
    durationCut = params['durationCut']
    durationRCosDecay = params['durationRCosDecay']

    if wavtemp.shape[0] > math.floor(durationCut * fs):
        wavtemp = wavtemp[:int(durationCut * fs)]
        wavtemp[wavtemp.shape[0] - int(fs * durationRCosDecay):] = wavtemp[
            wavtemp.shape[0] - int(
                fs * durationRCosDecay):] * utils.raised_cosine(
                    np.arange(int(fs * durationRCosDecay)), 0,
                    int(fs * durationRCosDecay))
    wavtemp = (wavtemp / 1.01) / (np.max(wavtemp) + np.finfo(float).eps)
    wavtemp = signal.resample(wavtemp, int(wavtemp.shape[0] / fs * new_fs))

    spectrogram__ = features.complexSpectrogram(wavtemp, windowSize, frameStep)
    spectrum__ = np.mean(
        np.abs(spectrogram__[:int(spectrogram__.shape[0] / 2), :]), axis=1)

    return spectrum__


def spectrogram(wavtemp, fs):
    # load analysis parameters
    params = load_params()
    new_fs = params['newFs']
    windowSize = params['windowSize']
    frameStep = params['frameStep']
    durationCut = params['durationCut']
    durationRCosDecay = params['durationRCosDecay']

    if wavtemp.shape[0] > math.floor(durationCut * fs):
        wavtemp = wavtemp[:int(durationCut * fs)]
        wavtemp[wavtemp.shape[0] - int(fs * durationRCosDecay):] = wavtemp[
            wavtemp.shape[0] - int(
                fs * durationRCosDecay):] * utils.raised_cosine(
                    np.arange(int(fs * durationRCosDecay)), 0,
                    int(fs * durationRCosDecay))
    wavtemp = (wavtemp / 1.01) / (np.max(wavtemp) + np.finfo(float).eps)
    wavtemp = signal.resample(wavtemp, int(wavtemp.shape[0] / fs * new_fs))

    spectrogram__ = features.complexSpectrogram(wavtemp, windowSize, frameStep)
    repres = np.abs(spectrogram__[:int(spectrogram__.shape[0] / 2), :])

    return repres


def mps(wavtemp, fs):
    # load analysis parameters
    params = load_params()
    new_fs = params['newFs']
    windowSize = params['windowSize']
    frameStep = params['frameStep']
    durationCut = params['durationCut']
    durationRCosDecay = params['durationRCosDecay']

    if wavtemp.shape[0] > math.floor(durationCut * fs):
        wavtemp = wavtemp[:int(durationCut * fs)]
        wavtemp[wavtemp.shape[0] - int(fs * durationRCosDecay):] = wavtemp[
            wavtemp.shape[0] - int(
                fs * durationRCosDecay):] * utils.raised_cosine(
                    np.arange(int(fs * durationRCosDecay)), 0,
                    int(fs * durationRCosDecay))
    wavtemp = (wavtemp / 1.01) / (np.max(wavtemp) + np.finfo(float).eps)
    wavtemp = signal.resample(wavtemp, int(wavtemp.shape[0] / fs * new_fs))
    wavtemp = np.r_[np.zeros(1000), wavtemp, np.zeros(1000)]

    spectrogram__ = features.complexSpectrogram(wavtemp, windowSize, frameStep)
    repres = np.transpose(
        np.abs(spectrogram__[:int(spectrogram__.shape[0] / 2), :]))

    N = repres.shape[0]
    M = repres.shape[1]
    # spatial, temporal zeros padding
    N1 = 2**utils.nextpow2(repres.shape[0])
    N2 = 2 * N1
    M1 = 2**utils.nextpow2(repres.shape[1])
    M2 = 2 * M1

    Y = np.zeros((N2, M2))

    # % first fourier transform (w.r.t. frequency axis)
    for n in range(N):
        R1 = np.abs(np.fft.fft(repres[n, :], M2))
        Y[n, :] = R1[:M2]

    # % second fourier transform (w.r.t. temporal axis)
    for m in range(M2):
        R1 = np.abs(np.fft.fft(Y[:N, m], N2))
        Y[:, m] = R1

    repres = np.abs(Y[:, :int(Y.shape[1] / 2)])
    # %scaleRateAngle = angle(Y) ;

    return repres


def strf(wavtemp, fs):
    # load analysis parameters
    params = load_params()
    new_fs = params['newFs']
    windowSize = params['windowSize']
    frameStep = params['frameStep']
    durationCut = params['durationCut']
    durationRCosDecay = params['durationRCosDecay']

    if wavtemp.shape[0] > math.floor(durationCut * fs):
        wavtemp = wavtemp[:int(durationCut * fs)]
        wavtemp[wavtemp.shape[0] - int(fs * durationRCosDecay):] = wavtemp[
            wavtemp.shape[0] - int(
                fs * durationRCosDecay):] * utils.raised_cosine(
                    np.arange(int(fs * durationRCosDecay)), 0,
                    int(fs * durationRCosDecay))
    wavtemp = (wavtemp / 1.01) / (np.max(wavtemp) + np.finfo(float).eps)
    wavtemp = signal.resample(wavtemp, int(wavtemp.shape[0] / fs * new_fs))
    wavtemp = np.r_[np.zeros(1000), wavtemp, np.zeros(1000)]

    spectrogram__ = features.complexSpectrogram(wavtemp, windowSize, frameStep)
    repres = np.transpose(
        np.abs(spectrogram__[:int(spectrogram__.shape[0] / 2), :]))

    N = repres.shape[0]
    M = repres.shape[1]
    # spatial, temporal zeros padding
    N1 = 2**utils.nextpow2(repres.shape[0])
    N2 = 2 * N1
    M1 = 2**utils.nextpow2(repres.shape[1])
    M2 = 2 * M1

    Y = np.zeros((N2, M2))

    # % first fourier transform (w.r.t. frequency axis)
    for n in range(N):
        R1 = np.abs(np.fft.fft(repres[n, :], M2))
        Y[n, :] = R1[:M2]

    # % second fourier transform (w.r.t. temporal axis)
    for m in range(M2):
        R1 = np.abs(np.fft.fft(Y[:N, m], N2))
        Y[:, m] = R1

    MPS_repres = np.abs(Y[:, :int(Y.shape[1] / 2)])

    # %% fourier strf
    maxRate = fs / frameStep / 2  #; % max rate values
    maxScale = windowSize / (fs * 1e-3) / 2  #; % max scale value
    ratesVector = np.linspace(-maxRate + 5, maxRate - 5, num=22)
    deltaRates = ratesVector[1] - ratesVector[0]
    scalesVector = np.linspace(0, maxScale - 5, num=11)
    deltaScales = scalesVector[2] - scalesVector[1]

    overlapRate = .75
    overlapScale = .75
    stdRate = deltaRates / 2 * (overlapRate + 1)
    stdScale = deltaScales / 2 * (overlapScale + 1)

    maxRatePoints = int(len(MPS_repres) / 2)
    maxScalePoints = MPS_repres.shape[1]
    stdRatePoints = maxRatePoints * stdRate / maxRate
    stdScalePoints = maxScalePoints * stdScale / maxScale

    # %STRF_repres = np.zeros((2*M, N, length(ratesVector), length(scalesVector)))
    STRF_repres = np.zeros((N, M, len(ratesVector), len(scalesVector)))
    # print('STRF_repres', STRF_repres.shape)
    for iRate in range(len(ratesVector)):
        rateCenter = ratesVector[iRate]
        # %rate center in point
        if rateCenter <= 0:
            rateCenterPoint = maxRatePoints * (
                2 - np.abs(rateCenter) / maxRate)
        else:
            rateCenterPoint = maxRatePoints * np.abs(rateCenter) / maxRate

        for iScale in range(len(scalesVector)):
            scaleCenter = scalesVector[iScale]
            # %scale center in point
            scaleCenterPoint = maxScalePoints * np.abs(scaleCenter) / maxScale
            filterPoint = gaussianWdw2d(rateCenterPoint, stdRatePoints,
                                          scaleCenterPoint, stdScalePoints,
                                          np.linspace(
                                              1,
                                              2 * maxRatePoints,
                                              num=2 * maxRatePoints),
                                          np.linspace(
                                              1,
                                              maxScalePoints,
                                              num=maxScalePoints))

            MPS_filtered = MPS_repres * filterPoint
            MPS_quadrantPoint = np.c_[MPS_filtered, np.fliplr(MPS_filtered)]
            stftRec = np.fft.ifft(np.transpose(np.fft.ifft(MPS_quadrantPoint)))
            ll = len(stftRec)
            stftRec = np.transpose(
                np.r_[stftRec[:M, :N], stftRec[ll - M:ll, :N]])
            # !! taking real values
            STRF_repres[:, :, iRate, iScale] = np.abs(stftRec[:, :int(stftRec.shape[1] / 2)])

    return STRF_repres


# def spectrogram(wavtemp, fs):
#     auditory_params = load_auditory_params()
#     new_fs = auditory_params['newFs']
#     durationCut = auditory_params['durationCut']
#     durationRCosDecay = auditory_params['durationRCosDecay']
#     if wavtemp.shape[0] > math.floor(durationCut * fs):
#         wavtemp = wavtemp[:int(durationCut * fs)]
#         wavtemp[wavtemp.shape[0] - int(fs * durationRCosDecay):] = wavtemp[
#             wavtemp.shape[0] - int(
#                 fs * durationRCosDecay):] * utils.raised_cosine(
#                     np.arange(int(fs * durationRCosDecay)), 0,
#                     int(fs * durationRCosDecay))
#     wavtemp = (wavtemp / 1.01) / (np.max(wavtemp) + np.finfo(float).eps)
#     wavtemp = signal.resample(wavtemp, int(wavtemp.shape[0] / fs * new_fs))
#     waveform2auditoryspectrogram_args = {
#         'frame_length': 1000 / 125,  # sample rate 125 Hz in the NSL toolbox
#         'time_constant': 8,
#         'compression_factor': -2,
#         'octave_shift': math.log2(new_fs / 16000),
#         'filt': 'p',
#         'VERB': 0
#     }
#     auditory_spec = features.waveform2auditoryspectrogram(
#         wavtemp, **waveform2auditoryspectrogram_args)
#     return auditory_spec

# def mps(wavtemp, fs):
#     auditory_params = load_auditory_params()
#     new_fs = auditory_params['newFs']
#     durationCut = auditory_params['durationCut']
#     durationRCosDecay = auditory_params['durationRCosDecay']
#     if wavtemp.shape[0] > math.floor(durationCut * fs):
#         wavtemp = wavtemp[:int(durationCut * fs)]
#         wavtemp[wavtemp.shape[0] - int(fs * durationRCosDecay):] = wavtemp[
#             wavtemp.shape[0] - int(
#                 fs * durationRCosDecay):] * utils.raised_cosine(
#                     np.arange(int(fs * durationRCosDecay)), 0,
#                     int(fs * durationRCosDecay))

#     wavtemp = (wavtemp / 1.01) / (np.max(wavtemp) + np.finfo(float).eps)
#     wavtemp = signal.resample(wavtemp, int(wavtemp.shape[0] / fs * new_fs))

#     waveform2auditoryspectrogram_args = {
#         'frame_length': 1000 / 125,  # sample rate 125 Hz in the NSL toolbox
#         'time_constant': 8,
#         'compression_factor': -2,
#         'octave_shift': math.log2(new_fs / 16000),
#         'filt': 'p',
#         'VERB': 0
#     }
#     stft = features.waveform2auditoryspectrogram(
#         wavtemp, **waveform2auditoryspectrogram_args)

#     strf_args = {
#         'num_channels': 128,
#         'num_ch_oct': 24,
#         'sr_time': 125,
#         'nfft_rate': 2 * 2**utils.nextpow2(stft.shape[0]),
#         'nfft_scale': 2 * 2**utils.nextpow2(stft.shape[1]),
#         'KIND': 2
#     }
#     # Spectro-temporal modulation analysis
#     # Based on Hemery & Aucouturier (2015) Frontiers Comp Neurosciences
#     # nfft_fac = 2  # multiplicative factor for nfft_scale and nfft_rate
#     # nfft_scale = nfft_fac * 2**utils.nextpow2(stft.shape[1])
#     mod_scale, phase_scale, _, _ = features.spectrum2scaletime(
#         stft, **strf_args)
#     # Scales vs. Time => Scales vs. Rates
#     repres, phase_scale_rate, _, _ = features.scaletime2scalerate(mod_scale * np.exp(1j * phase_scale),\
#                                                             **strf_args)
#     repres = repres[:, :int(repres.shape[1] / 2)]
#     return repres

# def strf(wavtemp, fs):

#     auditory_params = load_auditory_params()
#     scales = auditory_params['scales']
#     rates = auditory_params['rates']
#     durationCut = auditory_params['durationCut']
#     durationRCosDecay = auditory_params['durationRCosDecay']
#     new_fs = auditory_params['newFs']

#     if wavtemp.shape[0] > math.floor(durationCut * fs):
#         wavtemp = wavtemp[:int(durationCut * fs)]
#         wavtemp[wavtemp.shape[0] - int(fs * durationRCosDecay):] = wavtemp[
#             wavtemp.shape[0] - int(
#                 fs * durationRCosDecay):] * utils.raised_cosine(
#                     np.arange(int(fs * durationRCosDecay)), 0,
#                     int(fs * durationRCosDecay))

#     wavtemp = (wavtemp / 1.01) / (np.max(wavtemp) + np.finfo(float).eps)
#     wavtemp = signal.resample(wavtemp, int(wavtemp.shape[0] / fs * new_fs))

#     # Peripheral auditory model (from NSL toolbox)

#     # # compute spectrogram with waveform2auditoryspectrogram (from NSL toolbox), first f0 = 180 Hz
#     # num_channels = 128  # nb channels (128 ch. in the NSL toolbox)
#     # num_ch_oct = 24  # nb channels per octaves (24 ch/oct in the NSL toolbox)
#     # sr_time = 125  # sample rate (125 Hz in the NSL toolbox)

#     waveform2auditoryspectrogram_args = {
#         'frame_length': 1000 / 125,  # sample rate 125 Hz in the NSL toolbox
#         'time_constant': 8,
#         'compression_factor': -2,
#         'octave_shift': math.log2(new_fs / 16000),
#         'filt': 'p',
#         'VERB': 0
#     }
#     # frame_length = 1000 / sr_time  # frame length (in ms)
#     # time_constant = 8  # time constant (lateral inhibitory network)
#     # compression_factor = -2
#     # # fac =  0,  y = (x > 0), full compression, booleaner.
#     # # fac = -1, y = max(x, 0), half-wave rectifier
#     # # fac = -2, y = x, linear function
#     # octave_shift = math.log2(new_fs / 16000)  # octave shift
#     stft = features.waveform2auditoryspectrogram(
#         wavtemp, **waveform2auditoryspectrogram_args)

#     strf_args = {
#         'num_channels': 128,
#         'num_ch_oct': 24,
#         'sr_time': 125,
#         'nfft_rate': 2 * 2**utils.nextpow2(stft.shape[0]),
#         'nfft_scale': 2 * 2**utils.nextpow2(stft.shape[1]),
#         'KIND': 2
#     }
#     # Spectro-temporal modulation analysis
#     # Based on Hemery & Aucouturier (2015) Frontiers Comp Neurosciences
#     # nfft_fac = 2  # multiplicative factor for nfft_scale and nfft_rate
#     # nfft_scale = nfft_fac * 2**utils.nextpow2(stft.shape[1])
#     mod_scale, phase_scale, _, _ = features.spectrum2scaletime(
#         stft, **strf_args)

#     # Scales vs. Time => Scales vs. Rates
#     # nfft_rate = nfft_fac * 2**utils.nextpow2(stft.shape[0])
#     scale_rate, phase_scale_rate, _, _ = features.scaletime2scalerate(mod_scale * np.exp(1j * phase_scale),\
#                                                             **strf_args)
#     #num_channels, num_ch_oct, sr_time, nfft_rate, nfft_scale)
#     cortical_rep = features.scalerate2cortical(
#         stft, scale_rate, phase_scale_rate, scales, rates, **strf_args)
#     #num_ch_oct, sr_time, nfft_scale, nfft_rate, 2)
#     return cortical_rep

# if __name__ == "__main__":
#     audio, fs = utils.audio_data('/Users/baptistecaramiaux/Work/Projects/TimbreProject_Thoret/Code\ and\ data/timbreStudies/ext/sounds/Iverson1993Whole/01.W.Violin.aiff')
#     spectrum(audio, fs)
