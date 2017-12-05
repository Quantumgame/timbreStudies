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
test.audioRepres = {'FourierMPS'} ;
test.projection.type = 'local' ;
test.optimization.numLoops = 100000 ;
test.optimization.initMeanSigma = 10.0;
test.optimization.initVarSigma = 0.5;

                
% initialize sound path
folderContent = dir(test.soundFolder);
nbFreq = 128; % number of frequencies
idx = [];
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
                fprintf('Snd %i | size ', s);
                for kk = 1:length(size(audioData))
                    fprintf('%i ', size(audioData,kk));
                end
                fprintf('\n');
            end

            % dimension reduction
            config = test.projection;
            nbSounds = length(data) ;
            projectedData = [] ;

            if strcmp(test.audioRepres{k} , 'AuditorySTRF')
                for j = 1:nbSounds
                    audioRep_avgT = squeeze(mean(abs(data{j}),1)) ;
                    for iFrequency = 1:nbFreq 
                        [pcomps, allAuditorySpectrogramTemp, latent] = pca(squeeze(audioRep_avgT(iFrequency,:,:))) ;
                        cum_explained = cumsum(latent / sum(latent)); 
                        where = find(cum_explained>=0.99);
                        idx = [idx where(1)];
                    end
                end

            elseif ((strcmp(test.audioRepres{k} , 'AuditoryMPS')) || ...
                    (strcmp(test.audioRepres{k} , 'AuditorySpectrogram'))  || ...
                    (strcmp(test.audioRepres{k} , 'FourierMPS'))  || ...
                    (strcmp(test.audioRepres{k} , 'FourierSpectrogram')))
                for j = 1:nbSounds
                    [pcomps, allAuditorySpectrogramTemp, latent] = pca(data{j}') ;
                    cum_explained = cumsum(latent / sum(latent)); 
                    where = find(cum_explained>=0.99);
                    idx = [idx where(1)];
                end
            end 
            
        end
        
    end
end


