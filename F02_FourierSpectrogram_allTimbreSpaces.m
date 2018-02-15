timbreSpaceNames  = dir('./ext/sounds/') ;
nbSpaces = length(timbreSpaceNames)-3 ;

arguments.numLoops = 500000; % number of iterations

for iFolder = 4:nbSpaces+3
    
clearvars -except arguments iFolder nbSpaces timbreSpaceNames

% aff ext functions
addpath(genpath('./ext/')) ;
addpath(genpath('./lib/')) ;

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
durationCut = .2  ;
durationRCosDecay = .05 ;

% compute 4D-STRF of all files
spectrogramTab = struct([]) ;
for iFile = 1:nbSounds
    disp(strcat(num2str(iFile) , '...')) ;
    soundsList(iFile).name
    spectrogramTab{iFile} = F02_FourierSpectrogram(soundsList(iFile).name,durationCut,durationRCosDecay) ;
end

%%
% % load perceptual results in soundfiles folder
matDisFileName = sprintf('./ext/data/%s_dissimilarity_matrix.txt',timbreSpace);
matDis = load(matDisFileName);
meanMatDis = treshape(matDis,3) ; % up triangle of the dissimilarity matrix
MAT = zeros(nbSounds,nbSounds) ;

%% optimization
allSpectrogram = [] ;

for i = 1:nbSounds
    allSpectrogramTemp = pcaGlobal5(spectrogramTab{i} / max(max(spectrogramTab{i})),.01) ;
    size(allSpectrogramTemp);
    allSpectrogram = [allSpectrogram allSpectrogramTemp(:)] ;
end

arguments.realtimeLog = 0 ;
arguments.logFilename = 'Test' ;
arguments.initMeanSigma = 10.0;
arguments.initVarSigma = 0.5;
[sigmas, kernel, correlations] = kernel_optim(allSpectrogram, matDis, arguments);

save(strcat('optim_FourierSpectrogram_',timbreSpace,'.mat'));

end

