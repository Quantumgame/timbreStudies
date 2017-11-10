function [sigmas, kernel_v, correlations] = kernel_optim(x, target, arguments)
%%%     
%%%     x : features
%%%     target : dissimilarity matrix
%%%     direction : 1 (ascent) or -1 (descent)
%%%
%%%     Inspired by: 
%%%         K. Patil, D. Pressnitzer, S. Shamma, M. Elhilali. 
%%%         Music in our ears: The biological bases of musical timbre perception
%%%         Public Library of Science Computational Biology, Vol. 8(11), Nov 2012.

num_loops = arguments.numLoops;


[ndims,ninstrus] = size(x);

no_samples = ninstrus*(ninstrus-1)/2;

grad_corrfunc = zeros(ndims,1);
sigmas = arguments.initMeanSigma + arguments.initVarSigma * randn(ndims,1) ;

correlations = zeros(num_loops,1);

target_v = target(find(triu(ones(size(target)),1)));
mean_target = mean(target_v) ;
std_target = std(target_v) ;

for loop = 1:num_loops
    
    sigmas = sigmas - grad_corrfunc .* sigmas ;
    
    for i=1:ninstrus
        for j=i+1:ninstrus
            kernel(i,j) = exp(-1*sum( ((x(:,i)-x(:,j))./sigmas).^2 ) ) ;
            dkernel(i,j,:) = 2*kernel(i,j)*((x(:,i)-x(:,j)).^2)./(sigmas.^3) ;
        end
    end
    
    kernel_v = kernel(find(triu(ones(size(kernel)),1)));
    mean_kernel = mean(kernel_v);
    std_kernel = std(kernel_v);
    
    Jn = sum((kernel_v-mean_kernel).*(target_v-mean_target)) ;
    Jd = (no_samples-1)*std_target*std_kernel ;
    
    tic;
    
    for k=1:ndims
        dkernel_k(:,:) = dkernel(:,:,k);
        dkernel_k_v = dkernel_k(find(triu(ones(size(dkernel_k)),1)));
        dJn = sum(dkernel_k_v.*(target_v-mean_target));
        dJd = (no_samples-1)/(no_samples)*std_target/std_kernel*sum(dkernel_k_v.*(kernel_v-mean_kernel));
        grad_corrfunc(k)=(Jd*dJn - Jn*dJd)/(Jd^2);
    end
    
    
    correlations(loop) = Jn/Jd ;   
    
    %verbose
    if(mod(loop, 500)==0)
        fprintf('\t loop=%d | grad=%.6f | J=%.6f\n', loop, norm(grad_corrfunc,2), correlations(loop));
        if arguments.log == 1
            t = datetime('now') ;
            key = sprintf('%04i%02i%02i%02i%02i%02.0f',t.Year,t.Month,t.Day,t.Hour,t.Minute,t.Second) ;
            save(strcat('./logs/',arguments.logFilename,'_optimCorr_',key,'.mat'),...
                 'loop','sigmas','kernel_v','correlations');
        end
    end
    elapsed(loop) = toc;
end
% mean(elapsed)

