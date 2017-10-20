%%% Copyright (c) Baptiste Caramiaux, Etienne Thoret
%%% All rights reserved
%%%
%%% Function which generates a ripple sounds based on a linear frequency
%%% scaled spectrogram
%%% 
%%% f1 : minimum frequency (Hz)
%%% fN : maximum frequency (Hz)
%%% N : number of harmonics used to generate the noise
%%% omega : temporal modulation (cyc/kHz)
%%% w : temporal modulation (Hz)
%%% d : depth of the ripple (between 0 and 1)
%%% phi : phase at 0 for spectral modulation (Hz)
%%% duration : duration of the sound (s)


function y = generateLinearFreqRippleSound(f1, fN, N, omega, w, d, phi, duration, fs)


    %t = linspace(0,duration,floor(duration * fs)) ; % time vector
    t = (1:floor(duration * fs)) ; % time vector
    y = zeros(1,length(t)) ; % initialization of the output
    freq = linspace(f1,fN,N);
    %freq = f1*(fN/f1).^((linspace(1,N,N)-1)/(N-1)) ;         
    
    for ii = 1:N
        %xi = log(freq(ii)/f1) ;
        xi = freq(ii)*2e-3;
        ai = 1 + d  * sin(2*pi*(w/fs*t+omega*xi)+phi) ;
        y  = y + ai .* sin(2*pi*freq(ii)/fs*t + rand*2*pi) ; % ./ sqrt(freq(ii))
    end

    y = y ./ max(abs(y)) ;
       
end