function repres = A03_AuditoryMPS_250Hz(filename,durationCut,durationRCosDecay)
    % parameters

    
    scalesVector = [0.25 0.35 0.50 0.71 1.0 1.41 2.00 2.83 4.00 5.66 8.00] ;
    %ratesVector  = [-128 -90.5 -64 -45.3 -32 -22 -16 -11.3 -8 -5.8 -4 2 1 .5 .5 1 2 4.0 5.8 8.0 11.3 16.0 22.6 32.0 45.3 64.0 90.5 128.0] ;
    ratesVector = [-128 -90.5 -64 -45.3 -32 -22.6 -16 -11.3 -8 -5.70 -4 4 5.70 8 11.3 16 22.6 32 45.3 64 90.5 128] ;
      fs = 16000;          % sample rate
      [wavtemp, fs_wav] = audioread(filename) ;
      
    if length(wavtemp) > floor(durationCut*fs_wav)
        wavtemp = wavtemp(1:floor(durationCut*fs_wav)) ;
        wavtemp(end-floor(fs_wav*durationRCosDecay):end) = wavtemp(end-floor(fs_wav*durationRCosDecay):end) .* raisedCosine((0:floor(fs_wav*durationRCosDecay)),0,floor(fs_wav*durationRCosDecay))' ; 
    end
    
    wavtemp = double(wavtemp) ./ 1.01 / max(double(wavtemp)) ;
    wavtemp = resample(wavtemp, fs, fs_wav) ; % resample to 8000 Hz 

    % Peripheral auditory model (from NSL toolbox)
    dlmwrite('wavtemp.txt', wavtemp, 'precision',16)
    % compute spectrogram with wav2aud (from NSL toolbox), first f0 = 180 Hz
    nbChannels = 128 ;              % nb channels (128 ch. in the NSL toolbox)
    nbChOct = 24 ;                  % nb channels per octaves (24 ch/oct in the NSL toolbox)
    sr_time = 250 ;                 % sample rate (125 Hz in the NSL toolbox)
    frameLength = 1000 / sr_time ;  % frame length (in ms)
    timeConstant = 8 ;              % time constant (lateral inhibitory network)
    compressionfactor = -2 ;        % fac =  0,  y = (x > 0), full compression, booleaner.
                            % fac = -1, y = max(x, 0), half-wave rectifier
                            % fac = -2, y = x, linear function
    octaveShift = log2(fs/16000) ;  % octave shift                                 

    [stft] = wav2aud(wavtemp', [frameLength timeConstant  compressionfactor octaveShift]) ;

    % Spectro-temporal modulation analysis 
    % Based on Hemery & Aucouturier (2015) Frontiers Comp Neurosciences
    nfft_fac = 2 ; % multiplicative factor for nfft_scale and nfft_rate

    % 1) Spectrogram => Scales vs. Time

    nfft_scale = nfft_fac * 2^nextpow2(length(stft(1,:))) ;
    [modulationScale, phaseScale] = spec2scaletime(stft, nbChannels, nbChOct, sr_time, nfft_scale) ;

    % 2) Scales vs. Time => Scales vs. Rates

    nfft_rate = nfft_fac * 2^nextpow2(length(modulationScale(:,1))) ;
    repres = scaletime2scalerate(modulationScale .* exp(1i * phaseScale), nbChannels, nbChOct, sr_time, nfft_rate, nfft_scale) ;
    
end