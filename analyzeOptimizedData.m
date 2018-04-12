clear all;
clc ;

% initialize file path
optimizedDataFolder  = './ext/optimizedData/' ;
addpath(optimizedDataFolder) ;

representationType = {'AuditorySpectrum','AuditorySpectrogram', 'AuditoryMPS', 'AuditorySTRF', ...
                      'FourierSpectrum', 'FourierSpectrogram',  'FourierMPS'} ;

%representationType = {'AuditorySpectrogram_250Hz','AuditorySpectrum_250Hz','AuditorySTRF_250'} ;
                  
%results variables
corrs = zeros(length(representationType), 12) ;
pvals = zeros(length(representationType), 12) ;
fileNameOnly = [] ;
h = figure;
nReps = length(representationType);
for ri = 1:nReps
    fileList = dir(strcat(optimizedDataFolder,'optim_',representationType{ri}, '_*.mat')) ;
    nbFiles = length(fileList) ;
    fileNameOnly = [] ;
    disp(representationType{ri}) ;
    for fi = 1:nbFiles
       tempFileName = strsplit(fileList(fi).name,{'_','.mat'}) ;
       fileNameOnly = [fileNameOnly tempFileName(3)] ;
       load(fileList(fi).name) ;
       [corrs(ri,fi), pvals(ri,fi)] = corr(meanMatDis,kernel) ;
       
       subplot(nReps, nbFiles, (ri-1)*nbFiles+fi)
       plot(correlations);
       if (ri-1)*nbFiles+fi <= nbFiles
           title(fileNameOnly(end));
       end
       if mod((ri-1)*nbFiles+fi-1, nbFiles)==0
           ylabel(representationType{ri});
       end
       set(gca,'fontsize', 4);
       set(gca,'Xticklabel',[]);
       %subplot(nReps, nbFiles, (ri-1)*nbFiles+fi)
       %plot(sigmas);
       
       %title(strcat(representationType{ri},'_',fileNameOnly(end)));
    end
end

saveas(h, 'all_correlations.eps');
%pause;
%fileNameOnly = [fileNameOnly 'Patil2012' 'Grey1977' 'Grey1978'] ;

%% plot all timbre spaces

subplot(121)
plot(corrs(:,:)')
xticks(linspace(1,nbFiles,nbFiles))
xticklabels(fileNameOnly)
xtickangle(45)
axis([0 13 -1 .2])
subplot(122)
boxplot(squeeze(correlationTab(:,1,:)),'labels',fileNameOnly)

xtickangle(45)
axis([0 13 -1 .2])

%% plot all timbre spaces
figure
boxplot(squeeze(correlationTab(:,1,:))','labels',representationType)
xtickangle(45)
axis([0 8 -1 .2])


