clearvars;
clc ;

timbreSpaceNames  = dir('./ext/sounds/') ;
nbSpaces = length(timbreSpaceNames)-3 ;

arguments.numLoops = 500000; % number of iterations

for iFolder = 4:nbSpaces+3
    
clearvars -except arguments iFolder nbSpaces timbreSpaceNames

% load NSL Toolbox functions (http://www.isr.umd.edu/Labs/NSL/Software.htm)
addpath(genpath('./NSLfunctions/')); 
addpath('./ext/') ;
addpath(genpath('./lib/')) ;
% initialize sound path
timbreSpace = timbreSpaceNames(iFolder).name ;
soundPath = sprintf('./ext/sounds/%s/',timbreSpace);
ext = 'aiff' ;
addpath(soundPath) ;
soundsList = dir(strcat(soundPath, '*.',ext)) ;
nbSounds = length(soundsList) ;

% audio parameters
fs = 44100 ;

% waveform cut settings
durationCut = .3 ;
durationRCosDecay = .05 ;

% compute spectrum of all files
STRFTab = struct([]) ;
for iFile = 1:nbSounds
    disp(strcat(num2str(iFile) , '...')) ;
    filename = soundsList(iFile).name
    STRFTab{iFile} = abs(F04_FourierSTRF(filename,durationCut,durationRCosDecay)) ;
end

% load perceptual results in soundfiles folder

matDisFileName = sprintf('./ext/data/%s_dissimilarity_matrix.txt',timbreSpace);
matDis = load(matDisFileName);
meanMatDis = treshape(matDis,3) ; % up triangle of the dissimilarity matrix
MAT = zeros(nbSounds,nbSounds) ;

%% pca 
nbFreq = 743; % number of frequencies
allStrfProj = [] ;
pcaProjections = zeros(nbSounds, size(STRFTab{1},2), size(STRFTab{1},4), size(STRFTab{1},3));
normalizationCoefs = zeros(nbSounds);
for i = 1:nbSounds
    fprintf('PCA on sound %02i\n',i);
    strfProj = PCA_STRF(abs(STRFTab{i}),nbFreq) ;
    normalizationCoefs(i) = max(strfProj(:));
    strfProj = strfProj / max(strfProj(:)) ;
    allStrfProj = [allStrfProj strfProj(:)] ;
end

%% optimization
arguments.realtimeLog = 0 ;
arguments.logFilename = 'Test' ;
arguments.initMeanSigma = 10.0;
arguments.initVarSigma = 0.5;

[sigmas, kernel, correlations] = kernel_optim(allStrfProj, matDis, arguments) ;

save(strcat('optim_FourierMPS_',timbreSpace,'.mat'));

end

