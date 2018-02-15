% scaleRate2cortical transforms a scale vs. rate representation to the
% cortical representation
%  
% inputs parameters:
%  scaleRate        : modulus of the rate vs. scale reprsentation (rows:
%                     time, colums: scales)
%  phaseScaleRate   : phase of the rate vs. scale reprsentation
%  stft             : auditory 
%  scalesVector     : central scales of bandpass filerbank
%  ratesVector      : central rates of bandpass filerbank
%  nbChannels       : number of channels of the spectrogram
%  nbChOct          : number of channels per octave
%  sr_time          : sample rate
%  nfft_scale       : number of points of the DFT
%  KIND             : Kind of filter (1. Gabor function 2. Gaussian
%  function)
%
% output parameters:
%  corticalRepresentation : 4D cortical representation 
%                           (time, frequency, scale, rate)
%  

function [corticalRepresentation] = scaleRate2cortical(scaleRate, phaseScaleRate, stft, scalesVector, ratesVector, nbChOct, sr_time, nfft_scale, nfft_rate, KIND)


LgtRateVector  = length(ratesVector) ;  % length rate vector
LgtScaleVector = length(scalesVector) ; % length scale vector
LgtFreq        = length(stft(1,:)) ;    % length frequency
LgtTime        = length(stft(:,1)) ;    % length time

corticalRepresentation = zeros(length(stft(:,1)), length(stft(1,:)), LgtScaleVector, LgtRateVector) ;


    for j = 1 : LgtRateVector

        % temporal filter
        fc_rate = ratesVector(j)  ;
        
%         clc;
%         disp(num2str(fc_rate)) ;

        t = (0 : nfft_rate/2 - 1)' / sr_time * abs(fc_rate) ;
        h = sin(2 * pi * t) .* t.^2 .* exp(-3.5 * t) * abs(fc_rate) ; % same shape as in Chi et al. (2005)      
        h  = h - mean(h) ;
        
        
        STRF_rate0 = fft(h,  nfft_rate) ;
        A          = angle(STRF_rate0(1 : nfft_rate/2));  % 
        STRF_rate  = abs(STRF_rate0(1 : nfft_rate/2)) ;   % 
        STRF_rate  = STRF_rate / max(STRF_rate) ;
        STRF_rate  = STRF_rate .* exp(1i * A) ;
        
        
        % rate filtering modification                     
        STRF_rate                = [STRF_rate(1:nfft_rate/2); zeros(1,nfft_rate/2)'] ;
        STRF_rate(nfft_rate/2+1) = abs(STRF_rate(nfft_rate/2+2)) ;
        
        if(fc_rate < 0)            
            STRF_rate = [STRF_rate(1); conj(flipud(STRF_rate(2:nfft_rate)))] ;
        end
        
        %plot(abs(STRF_rate)); pause
        z1 = zeros(nfft_rate, nfft_scale/2); 
        for m = 1 : nfft_scale / 2
            z1(:, m) = STRF_rate .* scaleRate(:,m) .* exp(1i * phaseScaleRate(:, m));
        end
        
        z1 = ifft(z1(1:nfft_rate,:));
        
        %z1(5,10)
        %figure;imagesc(abs(z1));pause
        
        % scale filtering
        for i = 1 : LgtScaleVector
            
            fc_scale = scalesVector(i) ;
%             disp(num2str(fc_scale)) ;
            R1	= (0 : nfft_scale/2 - 1)'/ (nfft_scale/2) * nbChOct/2 / abs(fc_scale) ;	% length = L + 1 for now

            if KIND == 1	
               C1      = 1 / 2 / .3 / .3;
               STRF_scale = exp(-C1 * (R1-1).^2) + exp(-C1 * (R1+1).^2) ; % fft of the Gabor-like function 
            elseif KIND == 2		    
               R1	= R1 .^ 2;		 
               STRF_scale = R1 .* exp(1 - R1) ; 	% single-side filter / fft Gaussian-function
               %H	= H .^ .25;
            end
            
            z = zeros(LgtTime, nfft_scale/2) ;
            
            for n = 1 : LgtTime
                temp    = ifft(STRF_scale' .* z1(n, :), nfft_scale) ;
                z(n, :) = temp(1:nfft_scale/2) ;
            end
            
            %figure;plot(abs(STRF_scale)); pause;

            %figure;imagesc(abs(z));

            %figure;plot(STRF_scale);

            corticalRepresentation(: , :, i, j) = z(1 : LgtTime, 1 : LgtFreq) ;

       end

% Just to remind, Gabor-like function
% R2	= (0 : nfft_scale-1)'/ nfft_scale / 2 * nbChOct * abs(fc_scale) ;	% length = L + 1 for now
% R2 = R2 - max(R2) / 2;
% hs = (1 - R2.^2) .* exp(-R2.^2 / 2) * abs(fc_scale) ;
% STRF_scale0 = fft(hs, 2 * nfft_scale) ;
% STRF_scale0 = STRF_scale0 / max(STRF_scale0) ;
    

    end

end