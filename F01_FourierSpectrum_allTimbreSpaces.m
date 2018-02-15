timbreSpaceNames  = dir('./ext/sounds/') ;
nbSpaces = length(timbreSpaceNames)-3 ;

arguments.numLoops = 500000; % number of iterations

for iFolder = 4:nbSpaces

clearvars -except arguments iFolder nbSpaces timbreSpaceNames

% load NSL Toolbox functions (http://www.isr.umd.edu/Labs/NSL/Software.htm)
addpath(genpath('./NSLfunctions/')); 
addpath('./ext/') ;
addpath('./lib/') ;

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

% compute spectrum of all files
SPECTRUMTab = struct([]) ;
for iFile = 1:nbSounds
    disp(strcat(num2str(iFile) , '...')) ;
    filename = soundsList(iFile).name
    SPECTRUMTab{iFile} = abs(F01_FourierSpectrum(filename, durationCut, durationRCosDecay)) ;
end

%%
% load perceptual results in soundfiles folder

matDisFileName = sprintf('./ext/data/%s_dissimilarity_matrix.txt',timbreSpace);
matDis = load(matDisFileName);
meanMatDis = treshape(matDis,3) ; % up triangle of the dissimilarity matrix
MAT = zeros(nbSounds,nbSounds) ;

%% optimization
allSpectrum = [] ;

for i = 1:nbSounds
    allSpectrumTemp = SPECTRUMTab{i}' / max(SPECTRUMTab{i}) ;
    allSpectrum = [allSpectrum allSpectrumTemp] ;

end

arguments.logFilename = 'Test' ;
arguments.initMeanSigma = 10.0;
arguments.initVarSigma = 0.5;
arguments.realtimeLog = 0 ;
arguments.log = 0;
tic
[sigmas, kernel, correlations] = kernel_optim(allSpectrum, matDis, arguments) ;
duration = toc;
save(strcat('optim_FourierSpectrum_',timbreSpace,'.mat'));

end
