% generate serie of random ripple sounds
close all
clearvars

fs = 44100 ; % sample frequency
f1 = 50 ; % min frequency
fN = 8000 ; % max frequency
N = 1000 ; % number of harmonics
omega = .5 ; % central spectral modulation (cyc/kHz)
w = -10 ; % central temporal modulation (Hz)
d = 1 ; % depth
phi = 0 ; % phase at 0
duration = .5 ; % sound duration (seconds)

sdw  = 2 ; % standard deviation of rendomized temporal modulation (Hz)
sdOmega = 2 ;  % standard deviation of rendomized spectral modulation (cyc/kHz)
prefix = '02_' ;
nbSounds = 15 ; % number of random sounds

for i = 1:nbSounds
    wT = rand*(2*sdw-1) ;
    omegaT = rand*(2*sdOmega-1) ;
    y = generateLinearFreqRippleSound(f1, fN, N, omega+omegaT, w+wT, d, phi, duration, fs) ;
    filename = strcat(prefix,'ripple_',num2str(omegaT),'_',num2str(wT),'.wav') ;
    audiowrite(filename, y, fs) ;
end
    
    

