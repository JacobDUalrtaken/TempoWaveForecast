
inputData = [rand(10,1) 10*rand(10,1)-5];
outputData = rand(10,1);
opt = genfisOptions('GridPartition');
opt.NumMembershipFunctions = [3 5];%2 inputs
opt.InputMembershipFunctionType = ["gaussmf" "trimf"];
% Generate the FIS.
fis = genfis(inputData,outputData,opt);
[x,mf] = plotmf(fis,'input',1);
figure
subplot(2,1,1)
plot(x,mf)
xlabel('input 1 (gaussmf)')
legend('show');
[x,mf] = plotmf(fis,'input',2);
subplot(2,1,2)
plot(x,mf)
legend('show');
xlabel('input 2 (trimf)');
%% Generate FIS Using Subtractive Clustering
load clusterdemo.dat
inputData = clusterdemo(:,1:2);
outputData = clusterdemo(:,3);
opt = genfisOptions('FCMClustering','FISType','mamdani');
opt.NumClusters = 3;
%opt.Verbose = 0;
fis = genfis(inputData,outputData,opt);
showrule(fis)
[x,mf] = plotmf(fis,'input',1);
figure;
subplot(3,1,1)
plot(x,mf)
xlabel('Membership Functions for Input 1')
[x,mf] = plotmf(fis,'input',2);
subplot(3,1,2)
plot(x,mf)
xlabel('Membership Functions for Input 2')
[x,mf] = plotmf(fis,'output',1);
subplot(3,1,3)
plot(x,mf)
xlabel('Membership Functions for Output')
%% single-output:anfis
load fuzex1trnData.dat
fis1 = anfis(fuzex1trnData);

x = fuzex1trnData(:,1);
anfisOutput = evalfis(fis1,x);
figure;
plot(x,fuzex1trnData(:,2),'*r',x,anfisOutput,'.b')
legend('Training Data','ANFIS Output','Location','NorthWest')
% The ANFIS data does not match the training data well
%To improve the match:
%Increase the number of membership functions in the FIS structure to 4 
%Doing so adds fuzzy rules and tunable parameters to the system.
%Increase the number of training epochs.
opt = anfisOptions('InitialFIS',4,'EpochNumber',40);
opt.DisplayErrorValues = 0;
opt.DisplayStepSize = 0;
fis = anfis(fuzex1trnData,opt);
figure
anfisOutput = evalfis(fis,x);
plot(x,fuzex1trnData(:,2),'*r',x,anfisOutput,'.b');
legend('Training Data','ANFIS Output','Location','NorthWest');

%% eval
fis = readfis('tipper');
options = evalfisOptions('NumSamplePoints',50);
output = evalfis(fis,[2 1],options);

[output,fuzzifiedIn,ruleOut,aggregatedOut,ruleFiring] = evalfis(fis,[2 1]);
outputRange = linspace(fis.output.range(1),fis.output.range(2),length(aggregatedOut))'; 
figure;
plot(outputRange,aggregatedOut,[output output],[0 1])
xlabel('Tip')
ylabel('Output Membership')
legend('Aggregated output fuzzy set','Defuzzified output')

%%
datatable = readtable('G:\Phd\sp500\sp500_cp.csv');
clsprc = datatable.AdjClose;
index = (1:122)';
data = [index clsprc];

