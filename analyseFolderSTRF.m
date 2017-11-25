clear all;
clc ;

% load NSL Toolbox functions (http://www.isr.umd.edu/Labs/NSL/Software.htm)
addpath(genpath('./NSLfunctions/')); 
addpath('./ext/') ;
clear COCHBA ;
global COCHBA ; 
load aud24; % load cochlear filter coefficients

% initialize sound path
timbreSpace = 'grey1977' ;
soundPath = sprintf('./ext/sounds/%s/',timbreSpace);
ext = 'aiff' ;
addpath(soundPath) ;
soundsList = dir(strcat(soundPath, '*.',ext)) ;
nbSounds = length(soundsList) ;

% audio parameters
windowSize = 2048 ;
frameStep = 512 ;
fs = 44100 ;

% scale / rate analysis parameters
maxRate = fs / frameStep / 2 ; % max rate values
maxScale = windowSize / (fs * 1e-3) / 2 ; % max scale value

clkBegin = clock ;

% scales & rates vectors
% scalesVector = [0.25 0.35 0.50 0.71 1.0 1.41 2.00 2.83 4.00 5.66 8.00] ;
% ratesVector  = [-128 -90.5 -64 -45.3 -32 -22 -16 -11.3 -8 -5.8 -4 4.0 5.8 8.0 11.3 16.0 22.6 32.0 45.3 64.0 90.5 128.0] ;

scalesVector = [0.25 0.35 0.50 0.71 1.0 1.41 2.00 2.83 4.00 5.66 8.00] ;
ratesVector  = [-128 -90.5 -64 -45.3 -32 -22 -16 -11.3 -8 -5.8 -4 4.0 5.8 8.0 11.3 16.0 22.6 32.0 45.3 64.0 90.5 128.0] ;

% waveform cut settings
durationCut = .3 ;
durationRCosDecay = .05 ;

% compute 4D-STRF of all files
STRFTab = struct([]) ;
for iFile = 1:nbSounds
    disp(strcat(num2str(iFile) , '...')) ;
    soundsList(iFile).name
    STRFTab{iFile} = STRFmodelWithCut(soundsList(iFile).name, scalesVector, ratesVector, durationCut, durationRCosDecay) ;
end

%%
% % load perceptual results in soundfiles folder
% run(strcat(soundPath,'mainBehavioral.m')) ;
% matDis = mean(matDis, 3)' ; % perceptual dissimilarity matrix
matDisFileName = sprintf('./ext/data/%s_dissimilarity_matrix.txt',timbreSpace);
matDis = load(matDisFileName);
meanMatDis = treshape(matDis,3) ; % up triangle of the dissimilarity matrix
MAT = zeros(nbSounds,nbSounds) ;

%% pca 
nbFreq = 128; % number of frequencies
allStrfProj = [] ;
pcaProjections = zeros(nbSounds, size(STRFTab{1},2), size(STRFTab{1},4), size(STRFTab{1},3));
normalizationCoefs = zeros(nbSounds);
for i = 1:nbSounds
    fprintf('PCA on sound %02i\n',i);
    %[strfProj, pcaProjections(i,:,:,:)] = PCA_STRF(abs(STRFTab{i}),nbFreq) ;
    strfProj = PCA_STRF(abs(STRFTab{i}),nbFreq) ;
    normalizationCoefs(i) = max(strfProj(:));
    strfProj = strfProj / max(strfProj(:)) ;
    allStrfProj = [allStrfProj strfProj(:)] ;
end
%%
%tab_red = zeros(prod(size(STRFTab{1})),nbSounds);
%for iFile = 1:nbSounds
%    tab_red(:,iFile) = STRFTab{iFile}(:);
%end

%% optimization
arguments.numLoops = 200000;
arguments.logFilename = 'Test' ;
arguments.initMeanSigma = 10.0;
arguments.initVarSigma = 0.5;
arguments.log = 0;
[sigmas, kernel, correlations] = kernel_optim(allStrfProj, matDis, arguments);

%%
subplot(1,3,1)
plot(correlations)
subplot(1,3,2)
imagesc(meanMatDis)
subplot(1,3,3)
imagesc(kernel)
% hold on;
% plot(meanMatDis./std(meanMatDis),'k')
% plot(kernel./std(kernel),'-r')
% hold off;
drawnow;
%%
%save('optim_session1_iverson93whole.mat');
