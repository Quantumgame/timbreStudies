function [v5] = spec2wav_LIN(x, paras, filt)


if nargin < 3, filt = 'p'; end;

if (filt=='k')
   error('Please use wav2aud_fir function for FIR filtering!');
end

if (filt == 'p_o') load aud24_old;
else global COCHBA; end;

[L, M] = size(COCHBA);	% p_max = L - 2;

% octave shift, nonlinear factor, frame length, leaky integration
shft	= paras(4);			% octave shift

N = length(x(:,1)) ;
v5 = zeros(N, M-1);
%CF = 440 * 2 .^ ((-31:97)/24);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% last channel (highest frequency)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p	= real(conj(COCHBA(1, M)));
B	= real(conj(COCHBA((0:p)+2, M)));
A	= imag(conj(COCHBA((0:p)+2, M)));
%y1	= filter(B, A, x);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% All other channels
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for ch = (M-1):-1:1
    
    	p  = real((COCHBA(1, ch))) ;	    % order of ARMA filter
    	B  = real((COCHBA((0:p)+2, ch))) ;	% moving average coefficients
    	A  = imag((COCHBA((0:p)+2, ch))) ;	% autoregressive coefficients
    	y1 = filter(A , B , real(x(:, ch))) ; 
%            pause
%        plot(real(x(:, ch)))
%        pause
%           freqz(A  , B  ,512) ;        
%           pause
%          plot(abs(y1)) ;
%          pause

       v5(:, ch) = y1;
            
    end


end


