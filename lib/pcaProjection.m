function projectedData = pcaProjection(audioRepresentations, repName, config)
    
    nbSounds = length(audioRepresentations) ;
    projectedData = [] ;
    
    if strcmp(repName , 'spectroTemporalReceptiveField')
        nbFreq = 128; % number of frequencies
        %pcaProjections = zeros(nbSounds, size(audioRepresentations{1},2), size(audioRepresentations{1},4), size(audioRepresentations{1},3));
        normalizationCoefs = zeros(nbSounds);
        for i = 1:nbSounds
            audioRep_avgT = squeeze(mean(abs(audioRepresentations{i}),1)) ;
            audioRep_avgT_PCA = pcaGlobal5(squeeze(audioRep_avgT(1,:,:)),.01) ;  
            nbDim = length(audioRep_avgT_PCA) ;
            audioRep_avgT   = squeeze(mean(abs(audioRepresentations{i}),1)) ;
            audioRep_avgT_PCA = zeros(nbFreq,nbDim) ;
            pplComponents = zeros(size(audioRep_avgT,1), size(audioRep_avgT,3), size(audioRep_avgT,2));

            for iFrequency = 1:nbFreq 
                strf_soundi(iFrequency,:) = pcaGlobal5(squeeze(audioRep_avgT(iFrequency,:,:)),.01) ;
            end
            normalizationCoefs(i) = max(strf_soundi(:));
            strf_soundi = strf_soundi / max(strf_soundi(:)) ;
            projectedData = [projectedData strf_soundi(:)] ;
        end

    else
        
        switch config.type
            case 'local'
                for i = 1:nbSounds
                    %[pcomps, allAuditorySpectrogramTemp] = pca(AuditorySpectrogramTab{i}') ;
                    allAuditorySpectrogramTemp = pcaGlobal5(audioRepresentations{i}', 0.1) ;
                    projectedData = [projectedData allAuditorySpectrogramTemp(:)] ;
                end
            case 'global'
                pcaArray = [] ;
                for i = 1:nbSounds
                    pcaArray = [pcaArray ; audioRepresentations{i}] ;
                end
                princomps = pca(pcaArray) ;
                for i = 1:nbSounds
                    allAuditorySpectrogramTemp = audioRepresentations{i} * princomps' ;
                    projectedData = [projectedData allAuditorySpectrogramTemp(:)] ;
                end
            case 'none'
                for i = 1:nbSounds
                    allAuditorySpectrogramTemp = audioRepresentations{i} / max(max(audioRepresentations{i})) ;
                    projectedData = [projectedData allAuditorySpectrogramTemp(:)] ;
                end
            otherwise
                error('Not implemented') ;
        end
        
    end 
end