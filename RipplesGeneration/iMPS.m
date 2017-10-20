function [stfth] = iMPS(scaleRateAmp, scaleRateAngle, N, N2, M, M2)


stft  = zeros(N, M2) ;
stfth = zeros(N, M) ;

Y = scaleRateAmp .* exp(1i * scaleRateAngle) ;


% 2-D IFFT
for m = 1:M2
	R1 = ifft(Y(:, m));
	stft(:, m) = R1(1:N);
end

for n = 1:N
	R1 = ifft(stft(n, :), M2);
	stfth(n, :) = R1(1:M);
end


end
