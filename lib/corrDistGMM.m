function [correlations, matDistObj] = corrDistGMM(x, matDis, typeOfDistance)

[ndims,ninstrus] = size(x);

matDistObj = zeros(ninstrus, ninstrus) ;

switch typeOfDistance
    case 'euclidean'
        for iInstru = 1:ninstrus
            for jInstru = 1:ninstrus
                matDistObj(iInstru, jInstru) = sqrt(sum(sum((abs(x(:,iInstru)) - abs(x(:,jInstru))).^2))) ;                
            end 
        end
    case 'kl' % symmetric kullback leibler divergence 
        for iInstru = 1:ninstrus   
            for jInstru = 1:ninstrus
                matDistObj(iInstru, jInstru) = sum(abs(x(:,iInstru)) .* log10(abs(x(:,iInstru))./ abs(x(:,jInstru))) +...
                                                   abs(x(:,jInstru)) .* log10(abs(x(:,jInstru))./ abs(x(:,iInstru)))) ;
            end
        end
end


matDistObj = tril(matDistObj,-1);

correlations = corrcoef(treshape(matDistObj',3),treshape(matDis,3)) ;
correlations = correlations(2,1) ;

end