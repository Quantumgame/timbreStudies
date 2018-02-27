clear all;
clc ;

timbreSpaceNames  = dir('./ext/sounds/') ;
nbSpaces = length(timbreSpaceNames)-3 ;

%arguments.numLoops = 200; % number of iterations
correlationsPCA = zeros(1,12) ;
correlationsNonPCA = zeros(1,12) ;

representation = 'A_MPS' ;

for iFolder = 4:nbSpaces+3   
iFolder
clearvars -except arguments iFolder nbSpaces timbreSpaceNames correlationsPCA correlationsNonPCA representation

% load NSL Toolbox functions (http://www.isr.umd.edu/Labs/NSL/Software.htm)
addpath(genpath('./NSLfunctions/')); 
addpath('./ext/') ;
addpath('./lib/') ;

clear COCHBA ;
global COCHBA ; 
load aud24; % load cochlear filter coefficients

% initialize sound path
timbreSpace =  timbreSpaceNames(iFolder).name 
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

% waveform cut settings
durationCut = .3 ;
durationRCosDecay = .05 ;

% compute 4D-STRF of all files
Tab = struct([]) ;
for iFile = 1:nbSounds
    disp(strcat(num2str(iFile) , '...')) ;
    soundsList(iFile).name
    
    switch representation
        case 'A_Spectrum'
            Tab{iFile} = A01_AuditorySpectrum_250hz(soundsList(iFile).name, durationCut, durationRCosDecay) ;        
        case 'A_Spectrogram'
            Tab{iFile} = A02_AuditorySpectrogram_250hz(soundsList(iFile).name, durationCut, durationRCosDecay) ;
        case 'A_MPS'
            Tab{iFile} = A03_AuditoryMPS_250Hz(soundsList(iFile).name, durationCut, durationRCosDecay) ;
        case 'A_STRF'
            Tab{iFile} = A04_AuditorySTRF_250hz(soundsList(iFile).name, durationCut, durationRCosDecay) ;
    end
        
end

%%
% % load perceptual results in soundfiles folder
matDisFileName = sprintf('./ext/data/%s_dissimilarity_matrix.txt',timbreSpace);
matDis = load(matDisFileName);
meanMatDis = treshape(matDis,3) ; % up triangle of the dissimilarity matrix
MAT = zeros(nbSounds,nbSounds) ;

%% pca 
allStrfNonProj = [] ;
for i = 1:nbSounds
    allStrfNonProj = [allStrfNonProj Tab{i}(:)];
end


%% distance

tic;
[correlationsNonPCA(iFolder-3)] = corrDist(allStrfNonProj,matDis,'euclidean') ;

duration = toc;

end

correlationsPCA
correlationsNonPCA
