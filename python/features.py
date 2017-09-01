'''
Tools to compute STRF 
Copyright (c) Baptiste Caramiaux, Etienne Thoret
All rights reserved
'''
import numpy as np
import aifc
import math
from scipy import signal

#### TOOLS


def _raisedCosine(x, mu, s):
    return 1 / 2 / s * (1 + np.cos((x - mu) / s * math.pi)) * s


def _nextpow2(n):
    counter = 1
    while math.pow(2, counter) < n:
        counter += 1
    return counter


def _angle(compl_values):
    real_values = np.array([x.real for x in compl_values])
    imag_values = np.array([x.imag for x in compl_values])
    return np.arctan2(imag_values, real_values)


def _sigmoid(x, fac):
    '''
    Compute sigmoidal function
    '''
    y = x
    if fac > 0:
        y = 1. / (1. + np.exp(-y / fac))
    elif fac == 0:
        y = (y > 0)
    elif fac == -1:
        y = np.max(y, 0)
    elif fac == -3:
        raise ValueError('not implemented')
    return y


#### SCALE/ RATE


def spec2scaletime(stft, nbChannels, nbChOct, sr_time, nfft_scale):
    LgtTime = stft.shape[0]
    modulationScale = np.zeros((LgtTime, nfft_scale), dtype=complex)
    phaseScale = np.zeros((LgtTime, nfft_scale))
    # perform a FFT for each time slice
    for i in range(LgtTime):
        modulationScale[i, :] = np.fft.fft(stft[i, :], nfft_scale)
        phaseScale[i, :] = _angle(modulationScale[i, :])
    modulationScale = np.abs(modulationScale)  # modulus of the fft
    scales = np.linspace(0, nfft_scale + 1, nbChOct)
    times = np.linspace(0, modulationScale.shape[1] + 1, LgtTime / sr_time)
    return modulationScale, phaseScale, times, scales


def scaletime2scalerate(modulationScale, nbChannels, nbChOct, sr_time,
                        nfft_rate, nfft_scale):
    LgtScale = modulationScale.shape[1]
    scaleRate = np.zeros((nfft_rate, LgtScale), dtype=complex)
    phaseScaleRate = np.zeros((nfft_rate, LgtScale))
    for i in range(LgtScale):
        scaleRate[:, i] = np.fft.fft(modulationScale[:, i], nfft_rate)
        phaseScaleRate[:, i] = _angle(scaleRate[:, i])
    scaleRate = np.abs(scaleRate)
    rates = np.linspace(0, nfft_rate + 1, sr_time)
    scales = np.linspace(0, nfft_scale + 1, nbChOct)
    return scaleRate, phaseScaleRate, rates, scales


def scaleRate2cortical(scaleRate, phaseScaleRate, stft, scalesVector,
                       ratesVector, nbChOct, sr_time, nfft_scale, nfft_rate,
                       KIND):
    LgtRateVector = len(ratesVector)
    LgtScaleVector = len(scalesVector)  # length scale vector
    LgtFreq = stft.shape[1]
    LgtTime = stft.shape[0]
    corticalRepresentation = np.zeros(
        (LgtTime, LgtFreq, LgtScaleVector, LgtRateVector), dtype=complex)
    for j in range(LgtRateVector):
        fc_rate = ratesVector[j]
        t = np.arange(nfft_rate / 2) / sr_time * abs(fc_rate)
        h = np.sin(2 * math.pi * t) * np.power(t, 2) * np.exp(
            -3.5 * t) * abs(fc_rate)
        h = h - np.mean(h)
        STRF_rate0 = np.fft.fft(h, nfft_rate)
        A = _angle(STRF_rate0[:nfft_rate // 2])
        A[0] = 0.0  # instead of pi
        STRF_rate = np.absolute(STRF_rate0[:nfft_rate // 2])
        STRF_rate = STRF_rate / np.max(STRF_rate)
        STRF_rate = STRF_rate * np.exp(1j * A)
        # rate filtering modification
        # STRF_rate                = [STRF_rate(1:nfft_rate/2); zeros(1,nfft_rate/2)']
        STRF_rate.resize((nfft_rate, ))
        STRF_rate[nfft_rate // 2] = np.absolute(STRF_rate[nfft_rate // 2 + 1])

        if (fc_rate < 0):
            STRF_rate[1:nfft_rate] = np.matrix.conjugate(
                np.flipud(STRF_rate[1:nfft_rate]))

        z1 = np.zeros((nfft_rate, nfft_scale // 2), dtype=complex)
        for m in range(nfft_scale // 2):
            z1[:, m] = STRF_rate * scaleRate[:, m] * np.exp(
                1j * phaseScaleRate[:, m])

        # z1.resize((nfft_rate,nfft_rate))
        for i in range(nfft_scale // 2):
            z1[:, i] = np.fft.ifft(z1[:, i])
        # print(z1[10,:])

        for i in range(LgtScaleVector):
            fc_scale = scalesVector[i]
            R1 = np.arange(nfft_scale / 2) / (
                nfft_scale / 2) * nbChOct / 2 / abs(fc_scale)
            if KIND == 1:
                C1 = 1 / 2 / .3 / .3
                STRF_scale = np.exp(-C1 * np.power(R1 - 1, 2)) + np.exp(
                    -C1 * np.power(R1 + 1, 2))
            elif KIND == 2:
                R1 = np.power(R1, 2)
                STRF_scale = R1 * np.exp(1 - R1)
            z = np.zeros((LgtTime, nfft_scale // 2), dtype=complex)
            for n in range(LgtTime):
                temp = np.fft.ifft(STRF_scale * z1[n, :], nfft_scale)
                z[n, :] = temp[:nfft_scale // 2]
            corticalRepresentation[:, :, i, j] = z[:LgtTime, :LgtFreq]
    return corticalRepresentation


#### NLS lite


def wav2aud(x_, paras, filt='p', VERB=0):
    # Wav2Aud form NSL toolbox
    # http://www.isr.umd.edu/Labs/NSL/Software.htm
    #

    if (filt == 'k'):
        raise ValueError('Please use wav2aud_fir function for FIR filtering!')

    if (filt == 'p_o'):
        COCHBA = np.genfromtxt('COCHBA_aud24_old.txt', dtype=str)
    else:
        COCHBA = np.genfromtxt('COCHBA_aud24.txt', dtype=str)
    # convert str to complex (may be a better way...)
    COCHBA = np.asarray(
        [[complex(i.replace('i', 'j')) for i in COCHBA[row, :]]
         for row in range(len(COCHBA))])

    L, M = COCHBA.shape[0], COCHBA.shape[1]  # p_max = L - 2;
    L_x = len(x_)  # length of input

    # octave shift, nonlinear factor, frame length, leaky integration
    shft = paras[3]  # octave shift
    fac = paras[2]  # nonlinear factor
    L_frm = round(paras[0] * 2**(4 + shft))  # frame length (points)

    alph = math.exp(-1 / (paras[1] * 2**(4 + shft))) if paras[1] else 0

    # hair cell time constant in ms
    haircell_tc = 0.5
    beta = math.exp(-1 / (haircell_tc * 2**(4 + shft)))

    # get data, allocate memory for ouput
    N = math.ceil(L_x / L_frm)
    x = x_.copy()
    x.resize((N * L_frm, 1))  # zero-padding
    v5 = np.zeros((N, M - 1))

    #% last channel (highest frequency)
    p = COCHBA[0, M - 1].real
    B = COCHBA[np.arange(int(p) + 1) + 1, M - 1].real
    A = COCHBA[np.arange(int(p) + 1) + 1, M - 1].imag
    y1 = signal.lfilter(B, A, x, axis=0)
    y2 = _sigmoid(y1, fac)
    if (fac != -2):
        y2 = signal.lfilter([1.0], [1.0, -beta], y2, axis=0)
    y2_h = y2
    # % All other channels
    for ch in range((M - 2), -1, -1):
        # ANALYSIS: cochlear filterbank
        p = COCHBA[0, ch].real
        B = COCHBA[np.arange(int(p) + 1) + 1, ch].real
        A = COCHBA[np.arange(int(p) + 1) + 1, ch].imag
        y1 = signal.lfilter(B, A, x, axis=0)
        y2 = _sigmoid(y1, fac)
        # hair cell membrane (low-pass <= 4 kHz) ---> y2 (ignored for linear)
        if (fac != -2): y2 = signal.lfilter([1.0], [1.0, -beta], y2, axis=0)

        y3 = y2 - y2_h
        y2_h = y2
        # half-wave rectifier ---> y4
        y4 = np.maximum(y3, 0)

        # temporal integration window ---> y5
        if alph:  # leaky integration
            y5 = signal.lfilter([1.0], [1.0, -alph], y4, axis=0)
            v5[:, ch] = y5[L_frm * np.arange(1, N + 1) - 1].reshape(
                -1, )
        else:  # short-term average
            if (L_frm == 1):
                v5[:, ch] = y4
            else:
                v5[:, ch] = np.mean(y4.reshape(L_frm, N), axis=0)

    return v5


#### STRF


def STRF(filename, scalesVector, ratesVector, durationCut, durationRCosDecay):

    # loading a wav file (uncomment to use)
    fs = 8000
    aif = aifc.open(filename, 'r')

    fs_wav = aif.getframerate()
    wavtemp = aif.readframes(aif.getnframes())
    wavtemp = np.fromstring(wavtemp, np.short).byteswap() / 32767.0
    # print(len(wavtemp),math.floor(durationCut * fs_wav))
    if wavtemp.shape[0] > math.floor(durationCut * fs_wav):
        wavtemp = wavtemp[:int(durationCut * fs_wav)]
        wavtemp[wavtemp.shape[0] - int(fs_wav * durationRCosDecay):] = wavtemp[
            wavtemp.shape[0] - int(
                fs_wav * durationRCosDecay):] * _raisedCosine(
                    np.arange(int(fs_wav * durationRCosDecay)), 0,
                    int(fs_wav * durationRCosDecay))

    wavtemp = (wavtemp / 1.01) / np.max(wavtemp)
    wavtemp = signal.resample(wavtemp, int(
        wavtemp.shape[0] / fs_wav * fs))  # resample to 8000 Hz

    # Peripheral auditory model (from NSL toolbox)

    # compute spectrogram with wav2aud (from NSL toolbox), first f0 = 180 Hz
    nbChannels = 128  # nb channels (128 ch. in the NSL toolbox)
    nbChOct = 24  # nb channels per octaves (24 ch/oct in the NSL toolbox)
    sr_time = 125  # sample rate (125 Hz in the NSL toolbox)
    frame_length = 1000 / sr_time  # frame length (in ms)
    time_constant = 8  # time constant (lateral inhibitory network)

    compression_factor = -2
    # fac =  0,  y = (x > 0), full compression, booleaner.
    # fac = -1, y = max(x, 0), half-wave rectifier
    # fac = -2, y = x, linear function

    octave_shift = math.log2(fs / 16000)  # octave shift
    stft = wav2aud(wavtemp, \
                   [frame_length, time_constant, compression_factor, octave_shift])

    # Spectro-temporal modulation analysis
    # Based on Hemery & Aucouturier (2015) Frontiers Comp Neurosciences
    nfft_fac = 2  # multiplicative factor for nfft_scale and nfft_rate
    nfft_scale = nfft_fac * 2**_nextpow2(stft.shape[1])
    modulationScale, phaseScale, _, _ = spec2scaletime(
        stft, nbChannels, nbChOct, sr_time, nfft_scale)

    # Scales vs. Time => Scales vs. Rates
    nfft_rate = nfft_fac * 2**_nextpow2(stft.shape[0])
    scaleRate, phaseScaleRate, _, _ = scaletime2scalerate(
        modulationScale * np.exp(1j * phaseScale), nbChannels, nbChOct,
        sr_time, nfft_rate, nfft_scale)
    corticalRepresentation = scaleRate2cortical(
        scaleRate, phaseScaleRate, stft, scalesVector, ratesVector, nbChOct,
        sr_time, nfft_scale, nfft_rate, 2)


if __name__ == "__main__":
    scalesVector = [
        0.25, 0.35, 0.50, 0.71, 1.0, 1.41, 2.00, 2.83, 4.00, 5.66, 8.00
    ]
    ratesVector = [
        -128, -90.5, -64, -45.3, -32, -22, -16, -11.3, -8, -5.8, -4, 2, 1, .5,
        .5, 1, 2, 4.0, 5.8, 8.0, 11.3, 16.0, 22.6, 32.0, 45.3, 64.0, 90.5,
        128.0
    ]
    durationCut = .3
    durationRCosDecay = .05
    STRF('../ext/sounds/01.W.Violin.aiff', scalesVector, ratesVector,
         durationCut, durationRCosDecay)
