clc; clear all; close all;

%% load iverson93whole analysis (pca, kernels, etc.)
load 'optim_session1_iverson93whole' ;

%% decode sigmas and plot together with strfs

% choose snd (from 1 to 16)
snd = 5;

dec_tab = zeros(16,128,11,28) ;

% compute and plot all the 
figure;
for snd = 1 : 16

    % do same process before computing PCA in main script
    strf_sndi = abs(STRFTab{snd});
    strf_sndi = squeeze(mean(strf_sndi,1)); 

    % get projection matrices (1 per freq_i)
    pca_sndi_proj = squeeze(pcaProjections(snd,:,:,:));
    %pca_sndi_proj = 2*rand(128,28,11)-1 ;
    % decode sigmas w.r.t these proj matrices
    dec_strf_proj = decoder(sigmas, pca_sndi_proj);
    dec_tab(snd,:,:,:) = dec_strf_proj ;
    subplot(4,4,snd);imagesc(squeeze(mean(dec_strf_proj,1))) ;
    title(strcat('Sound',num2str(snd)));
end

figure;
for snd = 1 : 16

    % do same process before computing PCA in main script
    strf_sndi = abs(STRFTab{snd});
    strf_sndi = squeeze(mean(strf_sndi,1)); 
    
    pca_sndi_proj = squeeze(pcaProjections(snd,:,:,:));
    subplot(4,4,snd);imagesc(squeeze(mean(pca_sndi_proj,1))) ;
    title(strcat('Sound',num2str(snd)));
end

%%

% plot for each freq
figure;
for freqi = 1:size(strf_sndi,1)
    subplot(1,2,1);
    imagesc(squeeze(strf_sndi(freqi,:,:)));
    title(sprintf('STRF (avg in time, freq %i)',freqi));
    colorbar;
    subplot(1,2,2);
    imagesc(squeeze(dec_strf_proj(freqi, :, :)));
    title(sprintf('Decoded weights (freq %i)',freqi));
    colorbar; 
    pause;
end

plot average scale/rate decoded over frequencies


