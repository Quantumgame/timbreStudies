function [strfPCA, nbDim] = PCA_STRF(STRF,nbFreq)

strfT = squeeze(mean(STRF,1)) ;
strfPCA = pcaGlobal5(squeeze(strfT(1,:,:)),.01) ;  
nbDim = length(strfPCA) ;
strfPCA = zeros(nbFreq,nbDim) ;
strfT   = squeeze(mean(STRF,1)) ;

for iFrequency = 1:nbFreq 
    strfPCA(iFrequency,:) = pcaGlobal5(squeeze(strfT(iFrequency,:,:)),.01) ;     
end


end