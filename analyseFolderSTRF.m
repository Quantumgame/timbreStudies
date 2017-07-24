clear all;
clc ;

% load NSL Toolbox functions (http://www.isr.umd.edu/Labs/NSL/Software.htm)
addpath(genpath('./NSLfunctions/')); 

clear COCHBA ;
global COCHBA ; 
load aud24; % load cochlear filter coefficients

% initialize sound path
soundPath = '../sounds/' ;
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
ratesVector  = [-128 -90.5 -64 -45.3 -32 -22 -16 -11.3 -8 -5.8 -4 2 1 .5 .5 1 2 4.0 5.8 8.0 11.3 16.0 22.6 32.0 45.3 64.0 90.5 128.0] ;

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

% load perceptual results in soundfiles folder
run(strcat(soundPath,'mainBehavioral.m')) ;
matDis = mean(matDis, 3)' ; % perceptual dissimilarity matrix
meanMatDis = treshape(matDis,3) ; % up triangle of the dissimilarity matrix

MAT = zeros(nbSounds,nbSounds) ;

%% pca + distance (à compléter modifier corriger avec l'optimisation)
nbFreq = 128; % number of frequencies
tab_red = [] ;
for i = 1:nbSounds
    i
    redDim1 = PCA_STRF(abs(STRFTab{i}),nbFreq) ;    
    redDim1 = redDim1 / max(redDim1(:)) ;
    tab_red = [tab_red redDim1(:)] ;
    for j = 1:nbSounds
         redDim2 = PCA_STRF(abs(STRFTab{j}),nbFreq) ;   
         redDim2 = redDim2 / max(redDim2(:)) ;
         
         MAT(i,j) = sqrt(sum((redDim1(:)-redDim2(:)).^2)) ;
    end
end

%% compute correlation between perceptual results and computed distances
arrayMAT = treshape(tril(MAT, -1)',3) ;
[r_eucl, p_eucl] = corr(meanMatDis,arrayMAT,'type','pearson')  ;

%%
sigma = ones(length(redDim1(:)),1) ;
grad = zeros(length(redDim1(:)),1) ;
MAT2 = zeros(nbSounds,nbSounds) ;

for iOptim = 1:1000
    iOptim;
    sigma = sigma + grad ;
    grad = gradient_new(tab_red, matDis + matDis', sigma) ;

% distance optimisée
for i = 1:nbSounds
    i;
    for j = 1:nbSounds
         MAT2(i,j) = gk(tab_red(:,i),tab_red(:,j),sigma) ;
    end
end

arrayMAT2 = treshape(tril(MAT2, -1)',3) ;
[r_eucl, p_eucl] = corr(meanMatDis,arrayMAT2,'type','pearson')  ;
r_eucl

end
