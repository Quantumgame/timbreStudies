function [decoded_embeddings] = decoder(embeddings, projection_matrix)
    % embeddings should be (128 x 11, 1) 
    normalize = 1;
    embeddings_tmp = embeddings;
    if(normalize)
%         embeddings_tmp = embeddings_tmp / sum(embeddings_tmp);
%         embeddings_tmp = ones(size(embeddings_tmp,1)) - embeddings_tmp;
        embeddings_tmp = max(embeddings_tmp) - embeddings_tmp;
    end
    embeddings_tensored = zeros(128, 11);
    n_freqs = 128;
    for d = 1:11
        embeddings_tensored(:,d) = embeddings_tmp((d-1)*n_freqs+1:d*n_freqs);
    end
%     normalize = 0;
%     if(normalize)
%         embeddings_tensored = embeddings_tensored / sum(sum(embeddings_tensored));
%         embeddings_tensored = ones(128, 11) ./ embeddings_tensored;
% %         for d = 1:11
% %             embeddings_tensored(:,d) = embeddings_tensored(:,d) / sum(embeddings_tensored(:,d));
% %             embeddings_tensored(:,d) = 1.0 / embeddings_tensored(:,d);
% %         end
% %     end
%     plot(embeddings_tensored(:))
%     pause;
    decoded_embeddings = zeros(128, 11, 28);
    for f = 1:128
        tmp_mat = repmat(embeddings_tensored(f,:)',1,28) .* repmat(projection_matrix(f,:,1),11,1);
        decoded_embeddings(f,:,:) = tmp_mat ; %embeddings_tensored(f,:) * squeeze(projection_matrix(f,:,:))';
    end
    mins = zeros(128,1);
    maxs = zeros(128,1);
    for f = 1:128
        mins(f) = min(min(decoded_embeddings(f,:,:)));
        maxs(f) = max(max(decoded_embeddings(f,:,:)));
    end
    mi = min(mins);
    ma = max(mins);
    for f = 1:128
        decoded_embeddings(f,:,:) = (decoded_embeddings(f,:,:) - mi) / (ma - mi);
    end
end