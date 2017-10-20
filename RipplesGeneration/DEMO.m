%%% Demonstration of generation of ripple sounds as defined in Shamma (2001)
%%% Shamma, S. (2001). On the role of space and time in auditory processing. 
%%% Trends in Cognitive Sciences, 5, 340?348.
%%%
%%% See also http://www.srmathias.com/ripple-sounds/
%%%
%%% 


close all
clearvars

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


y = generateLinearFreqRippleSound(f1, fN, N, omega, w, d, phi, duration, fs) ;

filename = 'testRipple.wav' ;
audiowrite(filename, y, fs) ;

%% MPS 
% audio parameters
windowSize = 2048 ;
frameStep = 512 ;
fs = 44100 ;

% scale / rate analysis parameters
maxRate = fs / frameStep / 2 ; % max rate values
maxScale = windowSize / (fs * 1e-3) / 2 ; % max scale value

[stft_1, scaleRateAmp_1, scaleRateAngle_1] = MPS('testRipple.wav', fs, windowSize, frameStep) ;
MPS__ = scaleRateAmp_1 .* exp(1i * scaleRateAngle_1) ; % complex values of MPS

%% plot
MPS_plot = flipud([(MPS__(end/2+1:end,1:end/2)) ; (MPS__(1:end/2,1:end/2)) ]') ;
figure ;
imagesc((abs(MPS_plot))) ;
nbTicks = 40 ;
set(gca, 'XTick', linspace(1,length(MPS_plot(1,:)),nbTicks+1),'XtickLabel', linspace(-maxRate,maxRate,nbTicks+1)) 
set(gca, 'YTick', linspace(1,length(MPS_plot(:,1)),nbTicks+1),'YtickLabel', fliplr(linspace(0,maxScale,nbTicks+1))) 
