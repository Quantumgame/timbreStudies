function projectedData = pcaProjection(audioRepresentations, repName)
    
    nbSounds = length(audioRepresentations) ;
    projectedData = [] ;

    if strcmp(repName , 'AuditorySTRF')
        nbFreq = 128; % number of frequencies
        nbComp = 6;
        for i = 1:nbSounds
            audioRep_avgT = squeeze(mean(abs(audioRepresentations{i}),1)) ;
            for iFrequency = 1:nbFreq 
                [pcomps, allAuditorySpectrogramTemp] = pca(squeeze(audioRep_avgT(iFrequency,:,:))) ;
                allAuditorySpectrogramTemp = allAuditorySpectrogramTemp(:,1:nbComp);
                strf_soundi(iFrequency,:) = allAuditorySpectrogramTemp(:) ;
            end
            strf_soundi = strf_soundi / max(strf_soundi(:)) ;
            projectedData = [projectedData strf_soundi(:)] ;
        end

     elseif ((strcmp(test.audioRepres{k} , 'AuditoryMPS')) || ...
                    (strcmp(test.audioRepres{k} , 'AuditorySpectrogram'))  || ...
                    (strcmp(test.audioRepres{k} , 'FourierMPS'))  || ...
                    (strcmp(test.audioRepres{k} , 'FourierSpectrogram')))
        for i = 1:nbSounds
            [pcomps, allAuditorySpectrogramTemp] = pca(audioRepresentations{i}') ;
            allAuditorySpectrogramTemp = allAuditorySpectrogramTemp(:,1:10);
            projectedData = [projectedData allAuditorySpectrogramTemp(:)] ;
        end
    else
        for i = 1:nbSounds
            projectedData = [projectedData audioRepresentations(:)] ;
        end
    end 
end