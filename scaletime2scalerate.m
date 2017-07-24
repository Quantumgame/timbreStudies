% scaletime2scalerate transforms a scale vs. time representation into a scale vs. rate 
% representation. Each scale slice is fourrier transform wrt the time axis.
%  
% inputs parameters:
%  modulationScale  : scale vs. time representation (rows: time, columns:
%  scales)
%  nbChannels       : number of channels of the spectrogram
%  nbChOct          : number of channels per octave
%  sr_time          : sample rate
%  nfft_rate        : number of points of the DFT
%
% output parameters:
%  scaleRate        : modulus of the rate vs. scale reprsentation
%  phaseScaleRate   : phase of the rate vs. scale representation
%  rates            : rate axis (in Hz)"
%  scales           : scale axis (in Cycles/Octaves or Cycles/Hertz)

function [scaleRate, phaseScaleRate, rates, scales] = scaletime2scalerate(modulationScale, nbChannels, nbChOct, sr_time, nfft_rate, nfft_scale)

%disp('fft-rate slices...') ;

LgtScale = length(modulationScale(1,:)) ;

scaleRate      = zeros(nfft_rate, LgtScale) ;
phaseScaleRate = zeros(nfft_rate, LgtScale) ;

    for i = 1 : LgtScale
   
      scaleRate(:, i)      = fft(modulationScale(:, i), nfft_rate)  ; % complex values of the fft wrt time for the slice i   .* hanning(length(abs(modulationScale(:, i))))
      phaseScaleRate(:, i) = angle(scaleRate(:, i)) ;                 % phase of the fft wrt to time
    
    end

scaleRate = abs(scaleRate) ;

rates  = linspace(0, sr_time, nfft_rate) ;
scales = linspace(0, nbChOct, nfft_scale) ;

end