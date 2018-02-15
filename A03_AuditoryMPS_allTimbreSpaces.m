clear all;
clc ;

timbreSpaceNames  = dir('./ext/sounds/') ;
nbSpaces = length(timbreSpaceNames)-3 ;

arguments.numLoops = 200; % number of iterations

for iFolder = 4:nbSpaces+3

clearvars -except arguments iFolder nbSpaces timbreSpaceNames

% load NSL Toolbox functions (http://www.isr.umd.edu/Labs/NSL/Software.htm)
addpath(genpath('./NSLfunctions/')); 
addpath('./ext/') ;
addpath(genpath('./lib/')) ;
clear COCHBA ;
global COCHBA ; 
load aud24; % load cochlear filter coefficients
 
% initialize sound path
timbreSpace = timbreSpaceNames(iFolder).name
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
AuditoryMPSTab = struct([]) ;
for iFile = 1:nbSounds
    disp(strcat(num2str(iFile) , '...')) ;
    filename = soundsList(iFile).name
    AuditoryMPSTab{iFile} = abs(A03_AuditoryMPS(filename,durationCut,durationRCosDecay)) ;
end


%%
% load perceptual results in soundfiles folder

matDisFileName = sprintf('./ext/data/%s_dissimilarity_matrix.txt',timbreSpace);
matDis = load(matDisFileName);
meanMatDis = treshape(matDis,3) ; % up triangle of the dissimilarity matrix
MAT = zeros(nbSounds,nbSounds) ;

% %% pca 
% 
% allAuditoryMPSProj = [] ;
% pcaProjections = zeros(nbSounds, size(MPSTab{1},1), size(MPSTab{1},1));
% normalizationCoefs = zeros(nbSounds);
% 
% for i = 1:nbSounds
%     fprintf('PCA on sound %02i\n',i);
%     [MPSProj, pcaProjections(i,:,:)] = pcaGlobal5(abs(MPSTab{i}),.01) ;
%     normalizationCoefs(i) = max(MPSProj(:));
%     MPSProj = MPSProj / max(MPSProj(:)) ;
%     allAuditoryMPSProj = [allAuditoryMPSProj MPSProj(:)] ;
% end

%% optmization
allAuditoryMPS = [] ;

for i = 1:nbSounds
    allAuditoryMPSTemp = pcaGlobal5(AuditoryMPSTab{i} / max(max(AuditoryMPSTab{i})),.01) ;
    size(allAuditoryMPSTemp);
    allAuditoryMPS = [allAuditoryMPS allAuditoryMPSTemp(:)] ;
end

arguments.realtimeLog = 0 ;
arguments.logFilename = 'Test' ;
arguments.initMeanSigma = 10.0;
arguments.initVarSigma = 0.5;

tic;
[sigmas, kernel, correlations] = kernel_optim(allAuditoryMPS, matDis, arguments) ;
duration = toc;

save(strcat('optim_AuditoryMPS_',timbreSpace,'.mat'));

end


