function spectrogram__ = FourierSpectrogram(wavtemp, fs_wav)

    windowSize = 2048 ; 
    frameStep = 512 ;


    % parameters
    durationCut = .3 ;
    durationRCosDecay = .05 ;
    fs = 16000 ;
    nfft = 2^nextpow2(floor(durationCut * fs)) ;
      
    if length(wavtemp) > floor(durationCut*fs_wav)
        wavtemp = wavtemp(1:floor(durationCut*fs_wav)) ;
        wavtemp(end-floor(fs_wav*durationRCosDecay):end) = wavtemp(end-floor(fs_wav*durationRCosDecay):end) .* raisedCosine((0:floor(fs_wav*durationRCosDecay)),0,floor(fs_wav*durationRCosDecay))' ; 
	end

    wavtemp = double(wavtemp) ./ 1.01 / max(double(wavtemp)) ;
    wavtemp = resample(wavtemp, fs, fs_wav) ; % resample to fs_wav Hz 

    spectrogram__ =  ComplexSpectrogram(wavtemp, windowSize, frameStep) ;
    spectrogram__ = abs(spectrogram__(1:end/2,:)) ;

end