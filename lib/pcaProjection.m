function [projectedData, pcomps] = pcaProjection(audioRepresentations, repName)
    
    nbSounds = length(audioRepresentations) ;
    projectedData = [] ;

    if strcmp(repName , 'AuditorySTRF')
        nbFreq = 128; % number of frequencies
        nbComp = 6;
        pcomps = cell(nbSounds, nbFreq) ;
        for i = 1:nbSounds
            audioRep_avgT = squeeze(mean(abs(audioRepresentations{i}),1)) ;
            for iFrequency = 1:nbFreq 
                [pcomps{i,iFrequency}, allAuditorySpectrogramTemp] = pca(squeeze(audioRep_avgT(iFrequency,:,:))) ;
                allAuditorySpectrogramTemp = allAuditorySpectrogramTemp(:,1:nbComp);
                strf_soundi(iFrequency,:) = allAuditorySpectrogramTemp(:) ;
            end
            strf_soundi = strf_soundi / max(strf_soundi(:)) ;
            projectedData = [projectedData strf_soundi(:)] ;
        end

     elseif((strcmp(repName , 'AuditoryMPS')) || ...
            (strcmp(repName , 'AuditorySpectrogram'))  || ...
            (strcmp(repName , 'FourierMPS'))  || ...
            (strcmp(repName , 'FourierSpectrogram')))
        nbComp = 10 ;
        pcomps = cell(nbSounds) ;
        for i = 1:nbSounds
            [pcomps{i}, allAuditorySpectrogramTemp] = pca(audioRepresentations{i}') ;
            allAuditorySpectrogramTemp = allAuditorySpectrogramTemp(:,1:nbComp);
            projectedData = [projectedData allAuditorySpectrogramTemp(:)] ;
        end
    else
        pcomps = {};
        for i = 1:nbSounds
            projectedData = [projectedData audioRepresentations{i}(:)] ;
        end
    end 
end