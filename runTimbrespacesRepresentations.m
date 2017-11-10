close all; clear all; clc ;

% setup env
addpath(genpath('./NSLfunctions/')); 
addpath('./lib');
addpath('./ext/') ;
clear COCHBA ; global COCHBA ;
load aud24;
unix('mkdir -p logs') ;

% test config
test.soundFolder = './ext/sounds' ;
test.audioRepres = {'spectrum', ...
                    'auditorySpectrogram', ...
                    'spectroTemporalReceptiveField'} ;
test.projection.type = 'local' ;
test.optimization.numLoops = 100000 ;
test.optimization.initMeanSigma = 10.0;
test.optimization.initVarSigma = 0.5;

                
% initialize sound path
folderContent = dir(test.soundFolder);
for i = 1:length(folderContent)
    
    if folderContent(i).name(1) ~= '.'
        timbreSpaceName = folderContent(i).name ;
        fprintf('[%s]\n',timbreSpaceName) ;
        
        soundPath = sprintf('./ext/sounds/%s/',timbreSpaceName);
        addpath(soundPath) ;
        soundsList = dir(strcat(soundPath, '*.aiff')) ;
        nbSounds = length(soundsList) ;
    
        for k = 1:length(test.audioRepres)
            fprintf('  - %s\n',test.audioRepres{k}) ;
            
            % audio representation
            repProcess = str2func(test.audioRepres{k}) ;
            data = struct([]) ;
            for s = 1:nbSounds
                [audioData, fs] = audioread(soundsList(s).name) ;
                data{s} = repProcess(audioData, fs) ;
            end

            % dimension reduction
            projectedData = pcaProjection(data, test.audioRepres{k}, test.projection) ;

            % dissimilarity matrices
            matDisFileName = sprintf('./ext/data/%s_dissimilarity_matrix.txt',timbreSpaceName);
            matDis = load(matDisFileName);

            % optimization
            test.optimization.log = 1;
            test.optimization.logFilename = strcat(timbreSpaceName,'_',test.audioRepres{k});
            [sigmas, kernel, correlations] = kernel_optim(projectedData, matDis, test.optimization);
            
            % logging
            t = datetime('now') ;
            key = sprintf('%04i%02i%02i%02i%02i%02.0f',t.Year,t.Month,t.Day,t.Hour,t.Minute,t.Second) ;
            save(strcat('./logs/',strcat(timbreSpaceName,'_',test.audioRepres{k}),'_data_',key,'.mat'),...
                 'test', 'data', 'projectedData', 'matDis', 'sigmas', 'kernel', 'correlations');
            
        end
        
    end
end

