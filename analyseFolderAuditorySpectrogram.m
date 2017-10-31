clear all;
clc ;

% load NSL Toolbox functions (http://www.isr.umd.edu/Labs/NSL/Software.htm)
addpath(genpath('./NSLfunctions/')); 
addpath('./ext/') ;
clear COCHBA ;
global COCHBA ; 
load aud24; % load cochlear filter coefficients

% initialize sound path
timbreSpace = 'Iverson1993Whole' ;
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
    AuditorySpectrogramTab{iFile} = AuditorySpectrogrammodelWithCut(soundsList(iFile).name, durationCut, durationRCosDecay) ;
end

%%
% % load perceptual results in soundfiles folder
% run(strcat(soundPath,'mainBehavioral.m')) ;
% matDis = mean(matDis, 3)' ; % perceptual dissimilarity matrix
matDisFileName = sprintf('./ext/data/%s_dissimilarity_matrix.txt',timbreSpace);
matDis = load(matDisFileName);
meanMatDis = treshape(matDis,3) ; % up triangle of the dissimilarity matrix
MAT = zeros(nbSounds,nbSounds) ;

% %% pca 
% nbFreq = 128; % number of frequencies
% allAudSpecProj = [] ;
% pcaProjections = zeros(nbSounds, size(AuditorySpectrogramTab{1},2));
% normalizationCoefs = zeros(nbSounds);
% for i = 1:nbSounds
%     fprintf('PCA on sound %02i\n',i);
%     [AudSpecProj, pcaProjections(i,:,:,:)] = PCA_AUDSPEC(abs(AuditorySpectrogramTab{i}'),nbFreq) ;
%     normalizationCoefs(i) = max(AudSpecProj(:));
%     AudSpecProj = AudSpecProj / max(AudSpecProj(:)) ;
%     allAudSpecProj = [allAudSpecProj AudSpecProj(:)] ;
% end

%%
%tab_red = zeros(prod(size(STRFTab{1})),nbSounds);
%for iFile = 1:nbSounds
%    tab_red(:,iFile) = STRFTab{iFile}(:);
%end

%% optimization
allAuditorySpectrogram = [] ;

for i = 1:nbSounds
    allAuditorySpectrogramTemp = pca(AuditorySpectrogramTab{i}' / max(max(AuditorySpectrogramTab{i}))) ;
    allAuditorySpectrogram = [allAuditorySpectrogram allAuditorySpectrogramTemp(:)] ;
end

arguments.numLoops = 7500;
arguments.initMeanSigma = 10.0;
arguments.initVarSigma = 0.5;
[sigmas, kernel, correlations] = kernel_optim(allAuditorySpectrogram, matDis, arguments);

%%
subplot(1,3,1)
plot(correlations)
subplot(1,3,2)
imagesc(meanMatDis)
subplot(1,3,3)
imagesc(kernel)

%% visualisation sigma
figure
imagesc(reshape(sigmas,38,128)) 

% hold on;
% plot(meanMatDis./std(meanMatDis),'k')
% plot(kernel./std(kernel),'-r')
% hold off;
drawnow;
%%
save('optim_session1_iverson93whole.mat');
