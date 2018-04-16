'''
Various tools 
Copyright (c) Baptiste Caramiaux, Etienne Thoret
'''
import numpy as np
import math
from scipy import signal


def angle(compl_values):
    real_values = np.array([x.real for x in compl_values])
    imag_values = np.array([x.imag for x in compl_values])
    return np.arctan2(imag_values, real_values)

def my_filter(B,A,X):
    n = len(A)
    z = np.zeros(n)      # Creates zeros if input z is omitted
    b = B / A[0]  
    a = A / A[0]
    Y = np.zeros(X.shape)
    Y = Y.astype(np.float64)
    z = z.astype(np.float64)
    for m in range(len(Y)):
       Y[m] = b[0] * X[m] + z[0]
       for i in range(1,n):
          z[i - 1] = b[i] * X[m] + z[i] - a[i] * Y[m]
    # z = z[1:n - 1]
    return Y
    # # X is 1-D
    # num_len = len(B)
    # den_len = len(A)
    # # Z = np.zeros((X.shape[0]+num_len-1,1))
    # # Y = np.zeros((X.shape[0]+den_len-1,1))
    # Y = np.zeros((X.shape[0],1))
    # # Z[num_len-1:] = X
    # for i in range(X.shape[0]):
    #     if i==0:
    #         Y[0] = B[0]/A[0]*X[0]
    #     else:
    #         num = 0.0
    #         for p in range(num_len):
    #             if (i-p>=0):
    #                 num += B[p] * X[i-p]
    #                 print(i, p, i-p, B[p], X[i-p])
    #         den = 0.0
    #         for q in range(1,den_len):
    #             if (i-q-1>=0):
    #                 den += A[q] * Y[i-q-1]
    #                 # print(A[q])
    #         Y[i] = 1.0/A[0] * (num-den)
    #         print(num,den,Y[i])
    # return Y


def spec2scaletime(stft, nbChannels, nbChOct, sr_time, nfft_scale):
    ''' 
    spec2scaletime transforms a spectrograms into a time vs. scales (in
    cycles/hz or cycles/octaves) representation. Each time slice is fourrier
    transform wrt the frequency axis (in octaves or hertz).
    '''
    LgtTime = stft.shape[0]
    modulationScale = np.zeros((LgtTime, nfft_scale), dtype=complex)
    phaseScale = np.zeros((LgtTime, nfft_scale))
    # perform a FFT for each time slice
    for i in range(LgtTime):
        modulationScale[i, :] = np.fft.fft(stft[i, :], nfft_scale)
        phaseScale[i, :] = angle(modulationScale[i, :])
    modulationScale = np.abs(modulationScale)  # modulus of the fft
    scales = np.linspace(0, nfft_scale + 1, nbChOct)
    times = np.linspace(0, modulationScale.shape[1] + 1, LgtTime / sr_time)
    return modulationScale, phaseScale, times, scales

def scaletime2scalerate(modulationScale, nbChannels, nbChOct, sr_time, nfft_rate, nfft_scale):
    # scaletime2scalerate transforms a scale vs. time representation into a scale vs. rate 
    # representation. Each scale slice is fourrier transform wrt the time axis.
    #  
    # inputs parameters:
    #  modulationScale  : scale vs. time representation (rows: time, columns:
    #  scales)
    #  nbChannels       : number of channels of the spectrogram
    #  nbChOct          : number of channels per octave
    #  sr_time          : sample rate
    #  nfft_rate        : number of points of the DFT
    #
    # output parameters:
    #  scaleRate        : modulus of the rate vs. scale reprsentation
    #  phaseScaleRate   : phase of the rate vs. scale representation
    #  rates            : rate axis (in Hz)"
    #  scales           : scale axis (in Cycles/Octaves or Cycles/Hertz)
    # function [scaleRate, phaseScaleRate, rates, scales] = scaletime2scalerate(modulationScale, nbChannels, nbChOct, sr_time, nfft_rate, nfft_scale)
    LgtScale = modulationScale.shape[1]
    scaleRate      = np.zeros((nfft_rate, LgtScale), dtype=complex)
    phaseScaleRate = np.zeros((nfft_rate, LgtScale))
    for i in range(LgtScale):
      scaleRate[:, i] = np.fft.fft(modulationScale[:, i], nfft_rate)
      phaseScaleRate[:, i] = angle(scaleRate[:, i]) 
    scaleRate = np.abs(scaleRate)
    rates  = np.linspace(0, nfft_rate+1, sr_time)
    scales = np.linspace(0, nfft_scale+1, nbChOct)
    return scaleRate, phaseScaleRate, rates, scales

def scaleRate2cortical(scaleRate, phaseScaleRate, stft, scalesVector, ratesVector, nbChOct, sr_time, nfft_scale, nfft_rate, KIND):
    # % scaleRate2cortical transforms a scale vs. rate representation to the
    # % cortical representation
    # %  
    # % inputs parameters:
    # %  scaleRate        : modulus of the rate vs. scale reprsentation (rows:
    # %                     time, colums: scales)
    # %  phaseScaleRate   : phase of the rate vs. scale reprsentation
    # %  stft             : auditory 
    # %  scalesVector     : central scales of bandpass filerbank
    # %  ratesVector      : central rates of bandpass filerbank
    # %  nbChannels       : number of channels of the spectrogram
    # %  nbChOct          : number of channels per octave
    # %  sr_time          : sample rate
    # %  nfft_scale       : number of points of the DFT
    # %  KIND             : Kind of filter (1. Gabor function 2. Gaussian
    # %  function)
    # %
    # % output parameters:
    # %  corticalRepresentation : 4D cortical representation 
    # %                           (time, frequency, scale, rate)
    # %  
    # function [corticalRepresentation] = scaleRate2cortical(scaleRate, phaseScaleRate, stft, scalesVector, ratesVector, nbChOct, sr_time, nfft_scale, nfft_rate, KIND)
    LgtRateVector  = len(ratesVector) 
    LgtScaleVector = len(scalesVector) # length scale vector
    LgtFreq        = stft.shape[1]
    LgtTime        = stft.shape[0]
    corticalRepresentation = np.zeros((LgtTime, LgtFreq, LgtScaleVector, LgtRateVector), dtype=complex)
    for j in range(LgtRateVector):
        fc_rate = ratesVector[j]
        t = np.arange(nfft_rate/2)/sr_time * abs(fc_rate)
        h = np.sin(2 * math.pi * t) * np.power(t,2) * np.exp(-3.5 * t) * abs(fc_rate)
        h  = h - np.mean(h)

        STRF_rate0 = np.fft.fft(h,  nfft_rate)
        A = angle(STRF_rate0[:nfft_rate//2]) 
        A[0] = 0.0 # instead of pi
        STRF_rate  = np.absolute(STRF_rate0[:nfft_rate//2])
        STRF_rate  = STRF_rate / np.max(STRF_rate)
        STRF_rate  = STRF_rate * np.exp(1j * A)


        # rate filtering modification                     
        # STRF_rate                = [STRF_rate(1:nfft_rate/2); zeros(1,nfft_rate/2)']
        STRF_rate.resize((nfft_rate,))
        STRF_rate[nfft_rate//2] = np.absolute(STRF_rate[nfft_rate//2+1])
        
        if(fc_rate < 0):
            STRF_rate[1:nfft_rate] = np.matrix.conjugate(np.flipud(STRF_rate[1:nfft_rate]))

        z1 = np.zeros((nfft_rate, nfft_scale//2), dtype=complex)
        for m in range(nfft_scale // 2):
            z1[:, m] = STRF_rate * scaleRate[:,m] * np.exp(1j * phaseScaleRate[:, m])
        
        # z1.resize((nfft_rate,nfft_rate))
        for i in range(nfft_scale//2):
            z1[:,i] = np.fft.ifft(z1[:,i])
        # print(z1[10,:])

        for i in range(LgtScaleVector):
            fc_scale = scalesVector[i]
            R1 = np.arange(nfft_scale/2 )/ (nfft_scale/2) * nbChOct/2 / abs(fc_scale)
            if KIND == 1:
               C1 = 1 / 2 / .3 / .3;
               STRF_scale = np.exp(-C1 * np.power(R1-1,2)) + np.exp(-C1 * np.power(R1+1,2))
            elif KIND == 2:
               R1 = np.power(R1,2)
               STRF_scale = R1 * np.exp(1 - R1)
            z = np.zeros((LgtTime, nfft_scale//2), dtype=complex)
            for n in range(LgtTime):
                temp = np.fft.ifft(STRF_scale * z1[n, :], nfft_scale)
                z[n, :] = temp[:nfft_scale//2]
            corticalRepresentation[: , :, i, j] = z[:LgtTime, :LgtFreq]
    return corticalRepresentation


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
            v5[:, ch] = y5[L_frm * np.arange(1,N+1) - 1].reshape(-1,)
        else:  # short-term average
            if (L_frm == 1):
                v5[:, ch] = y4
            else:
                v5[:, ch] = np.mean(y4.reshape(L_frm, N), axis=0)

    return v5


if __name__ == "__main__":
    wav2aud()