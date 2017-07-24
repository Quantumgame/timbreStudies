%  spec2scaletime transforms a spectrograms into a time vs. scales (in
%  cycles/hz or cycles/octaves) representation. Each time slice is fourrier
%  transform wrt the frequency axis (in octaves or hertz).
% 
% inputs parameters:
%  stft         : spectrogram (rows: time, colums: octaves or Hz)
%  nbChannels   : number of channels of the spectrogram
%  nbChOct      : number of channels per octave
%  sr_time      : sample rate
%  nfft_scale   : number of points of the DFT
%
% output parameters:
%  modulationScale  : modulus of scale vs. time reprsentation
%  phaseScale       : phase of scale vs. time representation
%  times            : time axis (in s)
%  scales           : scale axis (in Cycle/Octave or Cycle/Hz)


function [modulationScale, phaseScale, times, scales] = spec2scaletime(stft, nbChannels, nbChOct, sr_time, nfft_scale)


%disp('fft-frequency slices...') ;

LgtTime = length(stft(:, 1)) ;

modulationScale = zeros(LgtTime , nfft_scale) ;
phaseScale      = zeros(LgtTime , nfft_scale) ;

    % perform a FFT for each time slice
    for i = 1 : LgtTime

       modulationScale(i, :) = fft(stft(i, :)  , nfft_scale)   ;  % complex values of the fft wrt frequency for the slice i  .* hanning(length(stft(i, :)))'
       phaseScale(i, :)      = angle(modulationScale(i, :)) ; % phase of the fft for the slice i
       
    end

modulationScale = abs(modulationScale) ; % modulus of the fft

scales = linspace(0, nbChOct, nfft_scale) ;
times  = linspace(0, LgtTime / sr_time, length(modulationScale(:,1))) ;


end