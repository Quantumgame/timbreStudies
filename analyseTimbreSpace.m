function [] = analyseTimbreSpace(timbreSpaceName)

    % setup env
    addpath(genpath('./NSLfunctions/')); 
    addpath('./lib');
    addpath('./ext/') ;
    clear COCHBA ; global COCHBA ;
    load aud24;
    unix('mkdir -p logs') ;

    % test config
    test.soundFolder = strcat('./ext/sounds/', timbreSpaceName) ;
    test.audioRepres = {'AuditorySpectrogram', ...
                        'AuditoryMPS', ...
                        'AuditorySpectrum', ...
                        'AuditorySTRF', ...
                        'FourierMPS', ...
                        'FourierSpectrogram', ...
                        'FourierSpectrum'} ;
    
    test.optimization.numLoops = 1000 ;
    test.optimization.initMeanSigma = 10.0;
    test.optimization.initVarSigma = 0.5;

    soundPath = sprintf('./ext/sounds/%s/',timbreSpaceName);
    addpath(soundPath) ;
    soundsList = dir(strcat(soundPath, '*.aiff')) ;
    nbSounds = length(soundsList) ;
    
    % looping on audio representations
    for k = 1:length(test.audioRepres)
        
        % logging
        t = datetime('now') ;
        key = sprintf('%04i%02i%02i-%02i%02i%02.0f',t.Year,t.Month,t.Day,t.Hour,t.Minute,t.Second);
        logFolderName = strcat('./logs/',timbreSpaceName,'_',test.audioRepres{k}, '_', key);
        logFileName = strcat(timbreSpaceName,'_',test.audioRepres{k});
        unix(sprintf('mkdir -p %s', logFolderName)) ;
        fprintf('- %s [%s]\n', test.audioRepres{k}, logFolderName) ;
        
        % audio representation
        repProcess = str2func(test.audioRepres{k}) ;
        data = struct([]) ;
        for s = 1:nbSounds
            [audioData, fs] = audioread(soundsList(s).name) ;
            data{s} = repProcess(audioData, fs) ;
        end

        % dimension reduction
        [projectedData, projectionMatrix] = pcaProjection(data, test.audioRepres{k}) ;

        % dissimilarity matrices
        matDisFileName = sprintf('./ext/data/%s_dissimilarity_matrix.txt',timbreSpaceName);
        matDis = load(matDisFileName);
        
        % optimization
        test.optimization.logFolderName = logFolderName;
        test.optimization.logFileName = logFileName;
        test.optimization.realtimeLog = 1;
        [sigmas, kernels, correlations, seed] = kernel_optim(projectedData, matDis, test.optimization);

        % saving
        save(strcat(logFolderName,'/', logFileName, '_', key, '_final.mat'),...
             'test', 'projectedData', 'projectionMatrix', 'matDis', 'sigmas', ...
             'kernels', 'correlations', 'seed');

    end
end

