% raised cosine function

function y = raisedCosine(x,mu,s)
    
    y = 1 / 2 / s * (1 + cos((x-mu) / s * pi)) * s ;

end