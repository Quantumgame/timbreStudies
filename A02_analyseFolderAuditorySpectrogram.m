clear all;
clc ;

% load NSL Toolbox functions (http://www.isr.umd.edu/Labs/NSL/Software.htm)
addpath(genpath('./NSLfunctions/')); 
addpath('./ext/') ;
clear COCHBA ;
global COCHBA ; 
load aud24; % load cochlear filter coefficients

% initialize sound path
timbreSpace = 'Patil2012_DX4' ;
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
% run(strcat(soundPath,'mainBehavioral.m')) ;
% matDis = mean(matDis, 3)' ; % perceptual dissimilarity matrix
matDisFileName = sprintf('./ext/data/%s_dissimilarity_matrix.txt',timbreSpace);
matDis = load(matDisFileName);
meanMatDis = treshape(matDis,3) ; % up triangle of the dissimilarity matrix
MAT = zeros(nbSounds,nbSounds) ;


% %% dimension reduction
% allAuditorySpectrogram = [] ;
% pca_config = 'global' ; % 'local', 'global', 'none'
% switch pca_config
%     case 'local'
%         for i = 1:nbSounds
%             %[pcomps, allAuditorySpectrogramTemp] = pca(AuditorySpectrogramTab{i}') ;
%             allAuditorySpectrogramTemp = pcaGlobal5(AuditorySpectrogramTab{i}', 100.0) ;
%             size(allAuditorySpectrogramTemp)
%             allAuditorySpectrogram = [allAuditorySpectrogram allAuditorySpectrogramTemp(:)] ;
%         end
%     case 'global'
%         pcaArray = [] ;
%         for i = 1:nbSounds
%             pcaArray = [pcaArray ; AuditorySpectrogramTab{i}] ;
%         end
%         princomps = pca(pcaArray) ;
%         for i = 1:nbSounds
%             allAuditorySpectrogramTemp = AuditorySpectrogramTab{i} * princomps' ;
%             allAuditorySpectrogram = [allAuditorySpectrogram allAuditorySpectrogramTemp(:)] ;
%         end
%     case 'none'
%         for i = 1:nbSounds
%             allAuditorySpectrogramTemp = AuditorySpectrogramTab{i} / max(max(AuditorySpectrogramTab{i})) ;
%             allAuditorySpectrogram = [allAuditorySpectrogram allAuditorySpectrogramTemp(:)] ;
%         end
%     otherwise
%         error('Not implemented') ;
% end


%% optimization
allAuditorySpectrogram = [] ;

for i = 1:nbSounds
    allAuditorySpectrogramTemp = pcaGlobal5(AuditorySpectrogramTab{i} / max(max(AuditorySpectrogramTab{i})),.01) ;
    size(allAuditorySpectrogramTemp);
    allAuditorySpectrogram = [allAuditorySpectrogram allAuditorySpectrogramTemp(:)] ;
end

arguments.numLoops = 200000;
arguments.logFilename = 'Test' ;
arguments.initMeanSigma = 10.0;
arguments.initVarSigma = 0.5;
arguments.realtimeLog = 0 ;
arguments.log = 0;
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
