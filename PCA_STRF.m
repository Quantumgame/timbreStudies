function [strfPCA, pplComponents] = PCA_STRF(STRF,nbFreq)

strfT = squeeze(mean(STRF,1)) ;
strfPCA = pcaGlobal5(squeeze(strfT(1,:,:)),.01) ;  
nbDim = length(strfPCA) ;
strfT   = squeeze(mean(STRF,1)) ;

strfPCA = zeros(nbFreq,nbDim) ;
pplComponents = zeros(size(strfT,1), size(strfT,3), size(strfT,2));

for iFrequency = 1:nbFreq 
    [strfPCA(iFrequency,:), pplComponents(iFrequency,:,:)] = pcaGlobal5(squeeze(strfT(iFrequency,:,:)),.01) ;
%     size(pc)
%     size(pplComponents)
%     pplComponents(iFrequency,:,:)=pc;
end


end