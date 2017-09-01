%% load NSL functions

function corticalRepresentation = STRFmodelWithCut(filename,scalesVector,ratesVector,durationCut,durationRCosDecay)




    %% loading a wav file (uncomment to use)
      fs = 8000;          % sample rate
      [wavtemp, fs_wav] = audioread(filename) ;
      
      if length(wavtemp) > floor(durationCut*fs_wav)
        wavtemp = wavtemp(1:floor(durationCut*fs_wav)) ;
        wavtemp(end-floor(fs_wav*durationRCosDecay):end) = wavtemp(end-floor(fs_wav*durationRCosDecay):end) .* raisedCosine((0:floor(fs_wav*durationRCosDecay)),0,floor(fs_wav*durationRCosDecay))' ; 
      

      %[wavtemp, fs_wav] = wavread(strcat(soundsPath, soundsList(iFile).name)) ;
      %wavtemp_44 = double(wavtemp) ./ 1.01 / max(double(wavtemp)) ;
      wavtemp = double(wavtemp) ./ 1.01 / max(double(wavtemp)) ;
      wavtemp = resample(wavtemp, fs, fs_wav) ; % resample to 8000 Hz 

    %% Peripheral auditory model (from NSL toolbox)
    dlmwrite('wavtemp.txt', wavtemp, 'precision',16)
    % compute spectrogram with wav2aud (from NSL toolbox), first f0 = 180 Hz
    nbChannels = 128 ;              % nb channels (128 ch. in the NSL toolbox)
    nbChOct = 24 ;                  % nb channels per octaves (24 ch/oct in the NSL toolbox)
    sr_time = 125 ;                 % sample rate (125 Hz in the NSL toolbox)
    frameLength = 1000 / sr_time ;  % frame length (in ms)
    timeConstant = 8 ;              % time constant (lateral inhibitory network)
    compressionfactor = -2 ;        % fac =  0,  y = (x > 0), full compression, booleaner.
                                    % fac = -1, y = max(x, 0), half-wave rectifier
                                    % fac = -2, y = x, linear function
    octaveShift = log2(fs/16000) ;  % octave shift                                 

    % rows: frequency, columns: time
    %[stft_mod, stft_phase] = wav2aud_LIN(wavtemp', [frameLength timeConstant  compressionfactor octaveShift]) ;
    % stft = stft_mod ;

    [stft] = wav2aud(wavtemp', [frameLength timeConstant  compressionfactor octaveShift]) ;
    

%     times     = linspace(0, length(stft(:,1)) / sr_time, length(stft(:,1))) ;
%     channels  = linspace(0, nbChannels, length(stft(1 , :))) ;

    % Spectro-temporal modulation analysis 
    % Based on Hemery & Aucouturier (2015) Frontiers Comp Neurosciences
    nfft_fac = 2 ; % multiplicative factor for nfft_scale and nfft_rate

    % 1) Spectrogram => Scales vs. Time

    nfft_scale = nfft_fac * 2^nextpow2(length(stft(1,:))) ;
    [modulationScale, phaseScale] = spec2scaletime(stft, nbChannels, nbChOct, sr_time, nfft_scale) ;
    
    
    % 2) Scales vs. Time => Scales vs. Rates

    nfft_rate = nfft_fac * 2^nextpow2(length(modulationScale(:,1))) ;
    [scaleRate, phaseScaleRate] = scaletime2scalerate(modulationScale .* exp(1i * phaseScale), nbChannels, nbChOct, sr_time, nfft_rate, nfft_scale) ;
    
    
    [corticalRepresentation] = scaleRate2cortical(scaleRate, phaseScaleRate, stft, scalesVector, ratesVector, nbChOct, sr_time, nfft_scale, nfft_rate, 2) ;
    corticalRepresentation(2,:,10,10)
    %size(corticalRepresentation)
    %plot(corticalRepresentation(2,:,10,10))
      else     
          corticalRepresentation = [] ;
      end
end

  