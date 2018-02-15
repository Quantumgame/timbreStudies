clearvars;
clc ;

% aff ext functions
addpath(genpath('./ext/')) ;
addpath(genpath('./lib/')) ;


% initialize sound path
timbreSpace = 'Grey1977' ;
soundPath = sprintf('./ext/sounds/%s/',timbreSpace);
ext = 'aiff' ;
addpath(soundPath) ;
soundsList = dir(strcat(soundPath, '*.',ext)) ;
nbSounds = length(soundsList) ;

% audio parameters
windowSize = 512 ;
frameStep = 128 ;
fs = 44100 ;

% waveform cut settings
durationCut = .2  ;
durationRCosDecay = .05 ;

% compute 4D-STRF of all files
spectrogramTab = struct([]) ;
for iFile = 1:nbSounds
    disp(strcat(num2str(iFile) , '...')) ;
    soundsList(iFile).name
    spectrogramTab{iFile} = FourierSpectrogram(soundsList(iFile).name,durationCut,durationRCosDecay) ;
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
allSpectrogram = [] ;

for i = 1:nbSounds
    allSpectrogramTemp = pcaGlobal5(spectrogramTab{i} / max(max(spectrogramTab{i})),.01) ;
    size(allSpectrogramTemp);
    allSpectrogram = [allSpectrogram allSpectrogramTemp(:)] ;
end

arguments.realtimeLog = 0 ;
arguments.logFilename = 'Test' ;
arguments.numLoops = 50000;
arguments.initMeanSigma = 10.0;
arguments.initVarSigma = 0.5;
[sigmas, kernel, correlations] = kernel_optim(allSpectrogram, matDis, arguments);

%%
subplot(1,3,1)
plot(correlations)
subplot(1,3,2)
imagesc(meanMatDis)
subplot(1,3,3)
imagesc(kernel)

%% visualisation sigma
figure
imagesc(reshape(sigmas,1024,8)) 

% hold on;
% plot(meanMatDis./std(meanMatDis),'k')
% plot(kernel./std(kernel),'-r')
% hold off;
drawnow;
%%
% save('optim_session1_iverson93whole.mat');
stftRec = abs([spectrogramTab{iFile}; flipud(spectrogramTab{iFile})]) ;

iterationCount = 1000 ; % number of iterations in the Griffin & Lim algorithm
dontEstimateTimeDelay = 0 ;
dontRandomizePhase = 0 ;
debugFlag = 0;

x = InvertSpectrogram(stftRec, frameStep, ...
    iterationCount, debugFlag, dontEstimateTimeDelay, dontRandomizePhase);
x = x / max(abs(x))   ; % normalize according to initial max
audiowrite('test.wav',x,16000);
