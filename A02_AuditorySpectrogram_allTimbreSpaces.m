timbreSpaceNames  = dir('./ext/sounds/') ;
nbSpaces = length(timbreSpaceNames)-3 ;

arguments.numLoops = 500000; % number of iterations

for iFolder = 4:nbSpaces+3
    
clearvars -except arguments iFolder nbSpaces timbreSpaceNames
    
    
% load NSL Toolbox functions (http://www.isr.umd.edu/Labs/NSL/Software.htm)
addpath(genpath('./NSLfunctions/')); 
addpath('./ext/') ;
addpath('./lib/') ;
clear COCHBA ;
global COCHBA ; 
load aud24; % load cochlear filter coefficients

% initialize sound path
timbreSpace = timbreSpaceNames(iFolder).name ;
soundPath = sprintf('./ext/sounds/%s/',timbreSpace);
ext = 'aiff' ;
addpath(soundPath) ;
soundsList = dir(strcat(soundPath, '*.',ext)) ;
nbSounds = length(soundsList) ;

% waveform cut settings
durationCut = .3 ;
durationRCosDecay = .05 ;

% compute 4D-STRF of all files
AuditorySpectrogramTab = struct([]) ;
for iFile = 1:nbSounds
    disp(strcat(num2str(iFile) , '...')) ;
    soundsList(iFile).name
    AuditorySpectrogramTab{iFile} = (A02_AuditorySpectrogram(soundsList(iFile).name, durationCut, durationRCosDecay)) ;
end

%%
% % load perceptual results in soundfiles folder

matDisFileName = sprintf('./ext/data/%s_dissimilarity_matrix.txt',timbreSpace);
matDis = load(matDisFileName);
meanMatDis = treshape(matDis,3) ; % up triangle of the dissimilarity matrix
MAT = zeros(nbSounds,nbSounds) ;

%% optimization
allAuditorySpectrogram = [] ;

for i = 1:nbSounds
    allAuditorySpectrogramTemp = pcaGlobal5(AuditorySpectrogramTab{i} / max(max(AuditorySpectrogramTab{i})),.01) ;
    size(allAuditorySpectrogramTemp)
    allAuditorySpectrogram = [allAuditorySpectrogram allAuditorySpectrogramTemp(:)] ;
end

arguments.logFilename = 'Test' ;
arguments.initMeanSigma = 10.0;
arguments.initVarSigma = 0.5;
arguments.realtimeLog = 0 ;
arguments.log = 0;
[sigmas, kernel, correlations] = kernel_optim(allAuditorySpectrogram, matDis, arguments);


save(strcat('optim_AuditorySpectrogram_',timbreSpace,'.mat'));

end


