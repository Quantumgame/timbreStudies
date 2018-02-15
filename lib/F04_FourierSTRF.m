function STRF_repres = F04_FourierSTRF(filename,durationCut,durationRCosDecay)

    % audio parameters
    fs = 16000 ;  % sample rate
    windowSize = 743 ;
    frameStep = 53 ;        
    
    % scale / rate analysis parameters
    maxRate = fs / frameStep / 2 ; % max rate values
    maxScale = windowSize / (fs * 1e-3) / 2 ; % max scale value
     
    ratesVector  = linspace(-maxRate+5,maxRate-5,22) ;
    deltaRates = ratesVector(2)-ratesVector(1) ;
    scalesVector = linspace(0,maxScale-5,11) ;
    deltaScales = scalesVector(2)-scalesVector(1) ;
  
    % scale / rate analysis parameters
    maxRate = fs / frameStep / 2 ; % max rate values
    maxScale = windowSize / (fs * 1e-3) / 2 ; % max scale value

    [wavtemp, fs_wav] = audioread(filename) ;
      
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

    MPS_repres = (Y(:,1:end/2)) ;
    
%% fourier strf   
    overlapRate  = .75 ;
    overlapScale = .75 ;
    stdRate  = deltaRates  / 2 * (overlapRate + 1) ;
    stdScale = deltaScales / 2 * (overlapScale + 1) ;

    maxRatePoints = length(MPS_repres(1:end/2,1)) ;
    maxScalePoints = length(MPS_repres(1,:)) ;
    stdRatePoints = maxRatePoints * stdRate / maxRate ;
    stdScalePoints = maxScalePoints * stdScale / maxScale ;

    %STRF_repres = zeros(2*M, N, length(ratesVector), length(scalesVector)) ;
    STRF_repres = zeros(N, M, length(ratesVector), length(scalesVector)) ;
    
    for iRate = 1:length(ratesVector)
        rateCenter = ratesVector(iRate) ;
        %rate center in point
        if rateCenter <= 0
            rateCenterPoint = maxRatePoints * (2 - abs(rateCenter) / maxRate) ;
        else
            rateCenterPoint = maxRatePoints * abs(rateCenter) / maxRate ;
        end

        for iScale = 1:length(scalesVector)
            scaleCenter = scalesVector(iScale) ;

            %scale center in point
            scaleCenterPoint = maxScalePoints * abs(scaleCenter) / maxScale ;
            [filterPoint] = gaussianWdw2d(rateCenterPoint, stdRatePoints,...
                                     scaleCenterPoint, stdScalePoints,...
                                     linspace(1,2*maxRatePoints,2*maxRatePoints),...
                                     linspace(1,maxScalePoints,maxScalePoints)) ;
            
            MPS_filtered = MPS_repres .* filterPoint ;
            MPS_quadrantPoint = [MPS_filtered fliplr(MPS_filtered)] ;
            stftRec = ifft(ifft(MPS_quadrantPoint)') ;
            stftRec = [stftRec(1:M,1:N); stftRec(end-M+1:end,1:N)]' ;
            STRF_repres(:, :, iRate, iScale) = stftRec(:,1:end/2) ;

        end
    
    end

end