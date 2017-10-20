clc
close all;
clearvars;

% audio parameters
windowSize = 2048 ;
frameStep = 256 ;
fs = 44100 ;

% Griffin & Lim
iterationCount = 100 ; % number of iterations in the Griffin & Lim algorithm
dontEstimateTimeDelay = 0 ;
dontRandomizePhase = 0 ;

% scale / rate analysis parameters
maxRate = fs / frameStep / 2 ; % max rate values
maxScale = windowSize / (fs * 1e-3) / 2 ; % max scale value


%% MPS 
filename1 = './VC_66.wav' ;
wavtemp = audioread(filename1) ;

[stft_1, scaleRateAmp_1, scaleRateAngle_1, N_1, N2_1, M_1, M2_1] = MPS(filename1, fs, windowSize, frameStep) ;
MPS__ = scaleRateAmp_1 .* exp(1i * scaleRateAngle_1) ; % complex values of MPS


 %% FILTER MPS HERE
% 

%MPS__ = abs(MPS__) ;

% just as an example, keep low rate only, DON'T FORGET THAT MPS__ is a four
% % quadrant
% MPS__(1:end-1,:) = 0 ; 
% 
% 
% subplot(121) ;
% imagesc(scaleRateAmp_1)
% subplot(122) ;
% imagesc(abs(MPS__)) ;

%% inverse MPS__
[stftR] = iMPS(abs(MPS__), angle(MPS__), N_1, N2_1, M_1, M2_1) ;

%% Reconstruction with the Griffin-Lim alogorithm
clkBegin = clock ;

stftRec = abs([stftR fliplr(stftR)]') ;

debugFlag = 0;

x = InvertSpectrogram(stftRec, frameStep, ...
    iterationCount, debugFlag, dontEstimateTimeDelay, dontRandomizePhase);
x = x / max(abs(x))   ; % normalize according to initial max

lgtComp = mod(clock - clkBegin, 60) ;
disp(strcat('Reconstruction Computation Duration = ',num2str(lgtComp(6)),'s')) ;

%x(isnan(x)) = 0;

%%
% figure;
% [b,a] = butter(2 , 100/fs,'high' ) ; % low-pass filter at 0.4 hz
% y     = filtfilt(b,a,x)  ; % remove DC_x (high-pass filter)
% plot(y)
% player = audioplayer([zeros(1,100) y], fs) ; % intialize player
% playblocking(player) ;

audiowrite('synth.wav',x,fs) ;
           
timerStartTest = tic ;


