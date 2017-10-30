function [AudSpecPCA, pplComponents] = PCA_AUDSPEC(AudSpec, nbFreq)

%AudSpecT = squeeze(mean(AudSpec,1)) ;
AudSpecPCA = pcaGlobal5(squeeze(AudSpecT(1,:)),.01) ;  
nbDim = length(AudSpecPCA) ;
AudSpecT   = squeeze(mean(AudSpec,1)) ;

AudSpecPCA = zeros(nbFreq,nbDim) ;
pplComponents = zeros(size(AudSpecT,1), size(AudSpecT,2));

for iFrequency = 1:nbFreq 
    [AudSpecPCA(iFrequency,:), pplComponents(iFrequency,:,:)] = pcaGlobal5(squeeze(AudSpecT(iFrequency,:,:)),.01) ;
%     size(pc)
%     size(pplComponents)
%     pplComponents(iFrequency,:,:)=pc;
end


end