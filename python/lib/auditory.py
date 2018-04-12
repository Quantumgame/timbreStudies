'''
Copyright (c) Baptiste Caramiaux, Etienne Thoret
All rights reserved

'''
import numpy as np
import math
from scipy import signal
from lib import utils
from lib import features  #import spectrum2scaletime, scaletime2scalerate, scalerate2cortical, waveform2auditoryspectrogram


def load_auditory_params():
    strf_params = {}
    strf_params['scales'] = [
        0.25, 0.35, 0.50, 0.71, 1.0, 1.41, 2.00, 2.83, 4.00, 5.66, 8.00
    ]
    strf_params['rates'] = [-128, -90.5, -64, -45.3, -32, -22, -16, -11.3, -8, -5.8, -4, 2, 1, .5, .5, \
                                   1, 2, 4.0, 5.8, 8.0, 11.3, 16.0, 22.6, 32.0, 45.3, 64.0, 90.5, 128.0]
    strf_params['durationCut'] = 0.3
    strf_params['durationRCosDecay'] = 0.05
    strf_params['newFs'] = 16000
    strf_params['sr_time'] = 250

    return strf_params


def spectrum(wavtemp, fs):
    auditory_params = load_auditory_params()
    new_fs = auditory_params['newFs']
    durationCut = auditory_params['durationCut']
    durationRCosDecay = auditory_params['durationRCosDecay']
    sr_time = auditory_params['sr_time']

    if wavtemp.shape[0] > math.floor(durationCut * fs):
        wavtemp = wavtemp[:int(durationCut * fs)]
        wavtemp[wavtemp.shape[0] - int(fs * durationRCosDecay):] = wavtemp[
            wavtemp.shape[0] - int(
                fs * durationRCosDecay):] * utils.raised_cosine(
                    np.arange(int(fs * durationRCosDecay)), 0,
                    int(fs * durationRCosDecay))
    wavtemp = (wavtemp / 1.01) / (np.max(wavtemp) + np.finfo(float).eps)
    wavtemp = signal.resample(wavtemp, int(wavtemp.shape[0] / fs * new_fs))
    waveform2auditoryspectrogram_args = {
        'frame_length': 1000 / sr_time,  # sample rate 125 Hz in the NSL toolbox
        'time_constant': 8,
        'compression_factor': -2,
        'octave_shift': math.log2(new_fs / 16000),
        'filt': 'p',
        'VERB': 0
    }
    auditory_spec = features.waveform2auditoryspectrogram(
        wavtemp, **waveform2auditoryspectrogram_args)
    return np.mean(auditory_spec, axis=0)


def spectrogram(wavtemp, fs):
    auditory_params = load_auditory_params()
    new_fs = auditory_params['newFs']
    durationCut = auditory_params['durationCut']
    durationRCosDecay = auditory_params['durationRCosDecay']
    sr_time = auditory_params['sr_time']

    if wavtemp.shape[0] > math.floor(durationCut * fs):
        wavtemp = wavtemp[:int(durationCut * fs)]
        wavtemp[wavtemp.shape[0] - int(fs * durationRCosDecay):] = wavtemp[
            wavtemp.shape[0] - int(
                fs * durationRCosDecay):] * utils.raised_cosine(
                    np.arange(int(fs * durationRCosDecay)), 0,
                    int(fs * durationRCosDecay))
    wavtemp = (wavtemp / 1.01) / (np.max(wavtemp) + np.finfo(float).eps)
    wavtemp = signal.resample(wavtemp, int(wavtemp.shape[0] / fs * new_fs))
    waveform2auditoryspectrogram_args = {
        'frame_length': 1000 / sr_time,  # sample rate 125 Hz in the NSL toolbox
        'time_constant': 8,
        'compression_factor': -2,
        'octave_shift': math.log2(new_fs / 16000),
        'filt': 'p',
        'VERB': 0
    }
    auditory_spec = features.waveform2auditoryspectrogram(
        wavtemp, **waveform2auditoryspectrogram_args)
    return auditory_spec


def mps(wavtemp, fs):
    auditory_params = load_auditory_params()
    new_fs = auditory_params['newFs']
    durationCut = auditory_params['durationCut']
    durationRCosDecay = auditory_params['durationRCosDecay']
    sr_time = auditory_params['sr_time']

    if wavtemp.shape[0] > math.floor(durationCut * fs):
        wavtemp = wavtemp[:int(durationCut * fs)]
        wavtemp[wavtemp.shape[0] - int(fs * durationRCosDecay):] = wavtemp[
            wavtemp.shape[0] - int(
                fs * durationRCosDecay):] * utils.raised_cosine(
                    np.arange(int(fs * durationRCosDecay)), 0,
                    int(fs * durationRCosDecay))

    wavtemp = (wavtemp / 1.01) / (np.max(wavtemp) + np.finfo(float).eps)
    wavtemp = signal.resample(wavtemp, int(wavtemp.shape[0] / fs * new_fs))

    waveform2auditoryspectrogram_args = {
        'frame_length': 1000 / sr_time,  # sample rate 125 Hz in the NSL toolbox
        'time_constant': 8,
        'compression_factor': -2,
        'octave_shift': math.log2(new_fs / 16000),
        'filt': 'p',
        'VERB': 0
    }
    stft = features.waveform2auditoryspectrogram(
        wavtemp, **waveform2auditoryspectrogram_args)

    strf_args = {
        'num_channels': 128,
        'num_ch_oct': 24,
        'sr_time': sr_time,
        'nfft_rate': 2 * 2**utils.nextpow2(stft.shape[0]),
        'nfft_scale': 2 * 2**utils.nextpow2(stft.shape[1]),
        'KIND': 2
    }
    # Spectro-temporal modulation analysis
    # Based on Hemery & Aucouturier (2015) Frontiers Comp Neurosciences
    # nfft_fac = 2  # multiplicative factor for nfft_scale and nfft_rate
    # nfft_scale = nfft_fac * 2**utils.nextpow2(stft.shape[1])
    mod_scale, phase_scale, _, _ = features.spectrum2scaletime(
        stft, **strf_args)
    # Scales vs. Time => Scales vs. Rates
    repres, phase_scale_rate, _, _ = features.scaletime2scalerate(mod_scale * np.exp(1j * phase_scale),\
                                                            **strf_args)
    repres = repres[:, :int(repres.shape[1] / 2)]
    return repres


def strf(wavtemp, fs):

    auditory_params = load_auditory_params()
    scales = auditory_params['scales']
    rates = auditory_params['rates']
    durationCut = auditory_params['durationCut']
    durationRCosDecay = auditory_params['durationRCosDecay']
    new_fs = auditory_params['newFs']
    sr_time = auditory_params['sr_time']
    
    if wavtemp.shape[0] > math.floor(durationCut * fs):
        wavtemp = wavtemp[:int(durationCut * fs)]
        wavtemp[wavtemp.shape[0] - int(fs * durationRCosDecay):] = wavtemp[
            wavtemp.shape[0] - int(
                fs * durationRCosDecay):] * utils.raised_cosine(
                    np.arange(int(fs * durationRCosDecay)), 0,
                    int(fs * durationRCosDecay))

    wavtemp = (wavtemp / 1.01) / (np.max(wavtemp) + np.finfo(float).eps)
    wavtemp = signal.resample(wavtemp, int(wavtemp.shape[0] / fs * new_fs))

    # Peripheral auditory model (from NSL toolbox)

    # # compute spectrogram with waveform2auditoryspectrogram (from NSL toolbox), first f0 = 180 Hz
    # num_channels = 128  # nb channels (128 ch. in the NSL toolbox)
    # num_ch_oct = 24  # nb channels per octaves (24 ch/oct in the NSL toolbox)
    # sr_time = 125  # sample rate (125 Hz in the NSL toolbox)

    waveform2auditoryspectrogram_args = {
        'frame_length': 1000 / sr_time,  # sample rate 125 Hz in the NSL toolbox
        'time_constant': 8,
        'compression_factor': -2,
        'octave_shift': math.log2(new_fs / 16000),
        'filt': 'p',
        'VERB': 0
    }
    # frame_length = 1000 / sr_time  # frame length (in ms)
    # time_constant = 8  # time constant (lateral inhibitory network)
    # compression_factor = -2
    # # fac =  0,  y = (x > 0), full compression, booleaner.
    # # fac = -1, y = max(x, 0), half-wave rectifier
    # # fac = -2, y = x, linear function
    # octave_shift = math.log2(new_fs / 16000)  # octave shift
    stft = features.waveform2auditoryspectrogram(
        wavtemp, **waveform2auditoryspectrogram_args)

    strf_args = {
        'num_channels': 128,
        'num_ch_oct': 24,
        'sr_time': sr_time,
        'nfft_rate': 2 * 2**utils.nextpow2(stft.shape[0]),
        'nfft_scale': 2 * 2**utils.nextpow2(stft.shape[1]),
        'KIND': 2
    }
    # Spectro-temporal modulation analysis
    # Based on Hemery & Aucouturier (2015) Frontiers Comp Neurosciences
    # nfft_fac = 2  # multiplicative factor for nfft_scale and nfft_rate
    # nfft_scale = nfft_fac * 2**utils.nextpow2(stft.shape[1])
    mod_scale, phase_scale, _, _ = features.spectrum2scaletime(
        stft, **strf_args)

    # Scales vs. Time => Scales vs. Rates
    # nfft_rate = nfft_fac * 2**utils.nextpow2(stft.shape[0])
    scale_rate, phase_scale_rate, _, _ = features.scaletime2scalerate(mod_scale * np.exp(1j * phase_scale),\
                                                            **strf_args)
    #num_channels, num_ch_oct, sr_time, nfft_rate, nfft_scale)
    cortical_rep = features.scalerate2cortical(
        stft, scale_rate, phase_scale_rate, scales, rates, **strf_args)
    #num_ch_oct, sr_time, nfft_scale, nfft_rate, 2)
    return cortical_rep


if __name__ == "__main__":
    audio, fs = utils.audio_data('/Users/baptistecaramiaux/Work/Projects/TimbreProject_Thoret/Code\ and\ data/timbreStudies/ext/sounds/Iverson1993Whole/01.W.Violin.aiff')
    spectrum(audio, fs)

