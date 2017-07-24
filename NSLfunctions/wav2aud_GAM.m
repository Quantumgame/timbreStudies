function [v5_mod, v5_phase] = wav2aud_GAM(x, paras, fs)
% WAV2AUD fast auditory spectrogramm (for band 180 - 7246 Hz)
%	v5 = wav2aud(x, [frmlen, tc, fac, shft], filt, VERB);
%	x	: the acoustic input.
%	v5	: the auditory spectrogram, N-by-(M-1) 
%
%	COCHBA  = (global) [cochead; cochfil]; (IIR filter)
%       cochead : 1-by-M filter length (<= L) vector.
%               f  = real(cochead); filter order
%               CF = imag(cochead); characteristic frequency
%	cochfil : (Pmax+2)-by-M (L-by-M) [M]-channel filterbank matrix.
%		B = real(cochfil); MA (Moving Average) coefficients.
%		A = imag(cochfil); AR (AutoRegressive) coefficients.
%	M	: highest (frequency) channel 
%
%	COCHBA  = [cochfil]; (IIR filter)
%	cochfil : (L-by-M) [M]-channel filterbank impulse responses.
%
%	PARAS	= [frmlen, tc, fac, shft];
%	frmlen	: frame length, typically, 8, 16 or 2^[natural #] ms.
%	tc	: time const., typically, 4, 16, or 64 ms, etc.
%		  if tc == 0, the leaky integration turns to short-term avg.
%	fac	: nonlinear factor (critical level ratio), typically, .1 for
%		  a unit sequence, e.g., X -- N(0, 1);
%		  The less the value, the more the compression.
%		  fac = 0,  y = (x > 0),   full compression, booleaner.
%		  fac = -1, y = max(x, 0), half-wave rectifier
%		  fac = -2, y = x,         linear function
%	shft	: shifted by # of octave, e.g., 0 for 16k, -1 for 8k,
%		  etc. SF = 16K * 2^[shft].%	
%
%	filt	: filter type, 'p'--> Powen's IIR filter (default)
%			       'p_o' --> Powen's old IIR filter (steeper group delay)	
%	
%	IIR filter : (24 channels/oct)
%	for the output of 	downsamp/shift	tc (64 ms)/ frame (16 ms)
%	==================================================================
%	180 - 7246		1	/0	1024	/ 256
%	90  - 3623		2	/-1	512	/ 128	*
%
%	Characteristic Frequency: CF = 440 * 2 .^ ((-31:97)/24);
%	Roughly, CF(60) = 1 (.5) kHz for 16 (8) kHz.
%
%	VERB	: verbose mode
%
%	WAV2AUD computes the auditory spectrogram for an acoustic waveform.
%	This function takes the advantage of IIR filter's fast performance
%	which not only reduces the computaion but also saves remarkable
%	memory space.
%	See also: AUD2WAV, UNITSEQ

% Auther: Powen Ru (powen@isr.umd.edu), NSL, UMD
% v1.00: 01-Jun-97

% Revision: Taishih Chi (tschi@isr.umd.edu), NSL, UMD
% v1.10: 04-Sep-98, add Kuansan's filter (as FIR filter)

% Revision: Taishih Chi (tschi@isr.umd.edu), NSL, UMD
% v2.00: 24-Jul-01, add hair cell membrane (lowpass) function

% Revision: Taishih Chi (tschi@isr.umd.edu), NSL, UMD
% v2.10: 04-Apr-04, remove FIR filtering option (see wav2aud_fir.m)

% get data, allocate memory for ouput 
x = hilbert(x(:)) ;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% All other channels
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for ch = (129-1):-1:1,
 
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% ANALYSIS: cochlear filterbank
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% (IIR) filter bank convolution ---> y1
    
    vec_fc = (-31:97) ;
    
    fc = floor(440 * 2 ^ (vec_fc(ch)/24)) / 2 ;
    
    [B] = gammatonefir(fc, fs, 300) ;
    
	y1 = filter(cell2mat(B), 1, x); 

    v5(:, ch) = y1;
            
end

v5_mod     = abs(v5) ;
v5_phase   = angle(v5) ;

end


