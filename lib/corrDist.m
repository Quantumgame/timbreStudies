function [correlations, matDistObj] = corrDist(x, matDis, arguments)

[ndims,ninstrus] = size(x);

matDistObj = zeros(ninstrus, ninstrus) ;

for iInstru = 1:ninstrus
    for jInstru = 1:ninstrus
        matDistObj(iIinstru, jInstru) = sqrt(sum(sum((x{iInstru} - x{jInstru}).^2))) ;
    end 
end

matDistObj = treshape(matDistObj,-1);

end