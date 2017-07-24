function [v5] = spec2wav_GAM(x, fs)

v5 = zeros(length(x(:,1)) , length(x(1,:))) ;

size(x)

    for ch = (129-1):-1:1,

    
      vec_fc = (-31:97) ;
    
      fc = floor(440 * 2 ^ (vec_fc(ch)/24)) / 2 ;
    
      [B] = gammatonefir(fc, fs, 300) ;
    
      y1 = filter(1, cell2mat(B), real(x(:, ch))); 
                 
      v5(:, ch) = y1;
            
    end


end


