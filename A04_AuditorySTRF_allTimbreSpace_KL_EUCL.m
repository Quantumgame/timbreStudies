clear all;
clc ;

timbreSpaceNames  = dir('./ext/sounds/') ;
nbSpaces = length(timbreSpaceNames)-3 ;

%arguments.numLoops = 200; % number of iterations
correlationsPCA = zeros(1,12) ;
correlationsNonPCA = zeros(1,12) ;
for iFolder = 4:nbSpaces+3
iFolder
clearvars -except arguments iFolder nbSpaces timbreSpaceNames correlationsPCA correlationsNonPCA

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
STRFTab = struct([]) ;
for iFile = 1:nbSounds
    disp(strcat(num2str(iFile) , '...')) ;
    soundsList(iFile).name
    STRFTab{iFile} = A04_AuditorySTRF_250hz(soundsList(iFile).name, durationCut, durationRCosDecay) ;
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
allStrfNonProj = [] ;
pcaProjections = zeros(nbSounds, size(STRFTab{1},2), size(STRFTab{1},4), size(STRFTab{1},3));
normalizationCoefs = zeros(nbSounds);
for i = 1:nbSounds
    fprintf('PCA on sound %02i\n',i);
    strfProj = PCA_STRF(abs(STRFTab{i}),nbFreq) ;
    normalizationCoefs(i) = max(strfProj(:));
    strfProj = strfProj / max(strfProj(:)) ;
    allStrfProj = [allStrfProj strfProj(:)] ;
    allStrfNonProj = [allStrfNonProj STRFTab{i}(:)];
end


%% distance

tic;
%[correlationsPCA(iFolder-3) ] = corrDist(allStrfProj,matDis,'kl') ;
[correlationsNonPCA(iFolder-3)] = corrDist(allStrfNonProj,matDis,'kl') ;

duration = toc;


end

correlationsPCA
correlationsNonPCA
