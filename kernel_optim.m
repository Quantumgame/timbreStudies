function [sigmas, kernel_v, correlations, seed] = kernel_optim(x, target, args)
%%%     kernel_optim
%%%     
%%%     x : features
%%%     target : dissimilarity matrix
%%%     direction : 1 (ascent) or -1 (descent)
%%%
%%%     Inspired by: 
%%%         K. Patil, D. Pressnitzer, S. Shamma, M. Elhilali. 
%%%         Music in our ears: The biological bases of musical timbre perception
%%%         Public Library of Science Computational Biology, Vol. 8(11), Nov 2012.

t = datetime('now') ;
key = sprintf('%04i%02i%02i%02i%02i%02.0f',t.Year,t.Month,t.Day,t.Hour,t.Minute,t.Second);


num_loops = args.numLoops;

[ndims,ninstrus] = size(x);

no_samples = ninstrus*(ninstrus-1)/2;
grad_corrfunc = zeros(ndims,1);

% init sigmas values
%sigmas = arguments.initMeanSigma + arguments.initVarSigma * randn(ndims,1) ;
load(strcat(args.logFolderName,'/seed.mat'));

correlations = zeros(num_loops,1);

target_v = target(find(triu(ones(size(target)),1)));
mean_target = mean(target_v) ;
std_target = std(target_v) ;

tic;

for loop = 1:num_loops
    
    for i=1:ninstrus
        for j=i+1:ninstrus
            x_i = x(:,i) ; %+ (max(x(:,i))/10.0)*rand(size(x,1), 1);
            x_j = x(:,j) ; %+ (max(x(:,j))/10.0)*rand(size(x,1), 1);
            kernel(i,j) = exp(-1*sum( ((x_i-x_j)./sigmas).^2 ) ) ;
            dkernel(i,j,:) = 2*kernel(i,j)*((x_i-x_j).^2)./(sigmas.^3) ;
        end
    end
    
    kernel_v = kernel(find(triu(ones(size(kernel)),1)));
    mean_kernel = mean(kernel_v);
    std_kernel = std(kernel_v);
    
    Jn = sum((kernel_v-mean_kernel).*(target_v-mean_target)) ;
    Jd = (no_samples-1)*std_target*std_kernel ;
    
    for k=1:ndims
        dkernel_k(:,:) = dkernel(:,:,k);
        dkernel_k_v = dkernel_k(find(triu(ones(size(dkernel_k)),1)));
        dJn = sum(dkernel_k_v.*(target_v-mean_target));
        dJd = (no_samples-1)/(no_samples)*std_target/std_kernel*sum(dkernel_k_v.*(kernel_v-mean_kernel));
        grad_corrfunc(k)=(Jd*dJn - Jn*dJd)/(Jd^2);
    end
    
    correlations(loop) = Jn/Jd ;   
    
    sigmas = abs(sigmas - grad_corrfunc .* sigmas) ;
    
    %verbose
    if(mod(loop, 200)==0)
        fprintf('  loop=%d | grad=%.6f | J=%.6f\n', loop, norm(grad_corrfunc,2), correlations(loop));
         if args.log
             elapsed = toc;
             save(strcat(args.logFolderName,'/',args.logFileName, '_', key,'_oldOptim.mat'), ...
                 'loop', 'elapsed', 'sigmas', 'kernel_v', 'correlations');
         end
    end
end

