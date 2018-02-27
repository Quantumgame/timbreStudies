function repres = F03_FourierMPS(filename,durationCut,durationRCosDecay)

   
    % audio parameters
      fs = 16000;          % sample rate
      windowSize = 743 ;
      frameStep = 186 ;        
      [wavtemp, fs_wav] = audioread(filename) ;
       wavtemp = padarray(wavtemp,16000,'post');
      
      if length(wavtemp) > floor(durationCut*fs_wav)
        wavtemp = wavtemp(1:floor(durationCut*fs_wav)) ;
        wavtemp(end-floor(fs_wav*durationRCosDecay):end) = wavtemp(end-floor(fs_wav*durationRCosDecay):end) .* raisedCosine((0:floor(fs_wav*durationRCosDecay)),0,floor(fs_wav*durationRCosDecay))' ; 
      end

    wavtemp = double(wavtemp) ./ 1.01 / max(double(wavtemp)) ;
    wavtemp = resample(wavtemp, fs, fs_wav) ; % resample to fs Hz 
      
    wavtemp = padarray(wavtemp, 1000) ; % zero-padding to remove intial and final artifacts
    stftA = ComplexSpectrogram(wavtemp, windowSize, frameStep);
    stft = abs(stftA(1:end/2,:))' ;

    [N,M] = size(stft) ;
    % spatial, temporal zeros padding 
    N1 = 2^nextpow2(N);	N2 = N1*2;
    M1 = 2^nextpow2(M);	M2 = M1*2;

    % first fourier transform (w.r.t. frequency axis)

    Y = zeros(N2, M2);

    for n = 1:N
        R1 = fft(stft(n, :), M2);
        Y(n, :) = R1(1:M2);
    end

    % second fourier transform (w.r.t. temporal axis)
    for m = 1:M2
        R1 = fft(Y(1:N, m), N2);
        Y(:, m) = R1;
    end

    repres = abs(Y(:,1:end/2)) ;
    %scaleRateAngle = angle(Y) ;
    
end