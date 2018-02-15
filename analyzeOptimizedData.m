clear all;
clc ;

% initialize file path
optimizedDataFolder  = './ext/data/optmizedData/' ;
representationType = {'AuditorySpectrum','AuditorySpectrogram', 'AuditoryMPS',...
                      'AuditorySTRF','FourierSpectrum','FourierSpectrogram',...
                      'FourierMPS'} ;

%representationType = {'AuditorySpectrogram_250Hz','AuditorySpectrum_250Hz','AuditorySTRF_250'} ;
                  
addpath(optimizedDataFolder) ;

%results variables
correlationTab = zeros(length(representationType),2,12) ;
fileNameOnly = [] ;

for iRepresentation = 1:length(representationType)
    fileList = dir(strcat(optimizedDataFolder,'optim_',representationType{iRepresentation}, '_*.mat')) ;
    nbFiles = length(fileList) ;
    fileNameOnly = [] ;
    disp(representationType{iRepresentation}) ;
    for iData = 1:nbFiles
       disp(iData) ;
       tempFileName = strsplit(fileList(iData).name,{'_','.mat'}) ;
       fileNameOnly = [fileNameOnly tempFileName(3)] ;
       load(fileList(iData).name) ;
       [correlationTab(iRepresentation,1,iData), correlationTab(iRepresentation,2,iData)] = ...
                        corr(meanMatDis,kernel) ;
                    
    %    tempFileName(3)
    %    correlationTab(1,iData)
    %    pause

    end
end

%fileNameOnly = [fileNameOnly 'Patil2012' 'Grey1977' 'Grey1978'] ;

%% plot all timbre spaces

subplot(121)
plot(squeeze(correlationTab(:,1,:))')
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


