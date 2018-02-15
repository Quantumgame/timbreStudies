clear all;
clc ;

% load NSL Toolbox functions (http://www.isr.umd.edu/Labs/NSL/Software.htm)
addpath(genpath('./NSLfunctions/')); 
addpath('./ext/') ;
addpath('./lib/') ;

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

% audio parameters
fs = 44100 ;

% waveform cut settings
durationCut = .3 ;
durationRCosDecay = .05 ;

% compute spectrum of all files
AuditorySpectrumTab = struct([]) ;
for iFile = 1:nbSounds
    disp(strcat(num2str(iFile) , '...')) ;
    filename = soundsList(iFile).name
    AuditorySpectrumTab{iFile} = abs(A01_AuditorySpectrum(filename, durationCut, durationRCosDecay)) ;
end

%%
% load perceptual results in soundfiles folder

matDisFileName = sprintf('./ext/data/%s_dissimilarity_matrix.txt',timbreSpace);
matDis = load(matDisFileName);
meanMatDis = treshape(matDis,3) ; % up triangle of the dissimilarity matrix
MAT = zeros(nbSounds,nbSounds) ;

% %% pca 
% 
% allStrfProj = [] ;
% pcaProjections = zeros(nbSounds, size(SPECTRUMTab{1},2));
% normalizationCoefs = zeros(nbSounds);
% for i = 1:nbSounds
%     fprintf('PCA on sound %02i\n',i);
%     [strfProj, pcaProjections(i,:,:,:)] = PCA_STRF(abs(STRFTab{i}),nbFreq) ;
%     normalizationCoefs(i) = max(strfProj(:));
%     strfProj = strfProj / max(strfProj(:)) ;
%     allStrfProj = [allStrfProj strfProj(:)] ;
% end

%%
%tab_red = zeros(prod(size(STRFTab{1})),nbSounds);
%for iFile = 1:nbSounds
%    tab_red(:,iFile) = STRFTab{iFile}(:); 
%end

%% optimization
allAuditorySpectrum = [] ;

for i = 1:nbSounds
    allAuditorySpectrumTemp = AuditorySpectrumTab{i} / max(AuditorySpectrumTab{i}) ;
    allAuditorySpectrum = [allAuditorySpectrum allAuditorySpectrumTemp'] ;

end

arguments.numLoops = 200000;
arguments.logFilename = 'Test' ;
arguments.initMeanSigma = 10.0;
arguments.initVarSigma = 0.5;
arguments.realtimeLog = 0 ;
arguments.log = 0;
[sigmas, kernel, correlations] = kernel_optim(allAuditorySpectrum, matDis, arguments) ;


% %%
% subplot(1,3,1)
% plot(correlations)
% subplot(1,3,2)
% imagesc(meanMatDis)
% subplot(1,3,3)
% imagesc(kernel)

% %% visualisation sigma
% figure;
% plot(sigmas);
% 
% % hold on;
% % plot(meanMatDis./std(meanMatDis),'k')
% % plot(kernel./std(kernel),'-r')
% % hold off;
% drawnow;
%%
save(strcat('optim_AuditorySpectrum.mat',timbreSpace,'.mat'));
