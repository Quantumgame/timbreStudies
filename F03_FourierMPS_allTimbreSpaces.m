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
MPSTab = struct([]) ;
for iFile = 1:nbSounds
    disp(strcat(num2str(iFile) , '...')) ;
    filename = soundsList(iFile).name
    MPSTab{iFile} = abs(F03_FourierMPS(filename,durationCut,durationRCosDecay)) ;
end

% load perceptual results in soundfiles folder

matDisFileName = sprintf('./ext/data/%s_dissimilarity_matrix.txt',timbreSpace);
matDis = load(matDisFileName);
meanMatDis = treshape(matDis,3) ; % up triangle of the dissimilarity matrix
MAT = zeros(nbSounds,nbSounds) ;

%% optimization
allMPS = [] ;

for i = 1:nbSounds
    allMPSTemp = pcaGlobal5(MPSTab{i} / max(max(MPSTab{i})),.01) ;
    allMPS = [allMPS allMPSTemp] ;
end

arguments.realtimeLog = 0 ;
arguments.logFilename = 'Test' ;
arguments.initMeanSigma = 10.0;
arguments.initVarSigma = 0.5;

[sigmas, kernel, correlations] = kernel_optim(allMPS, matDis, arguments) ;

save(strcat('optim_FourierMPS_',timbreSpace,'.mat'));

end

