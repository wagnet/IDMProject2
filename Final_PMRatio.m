%% Script to help get started on DDI Final Project


%% Input Data 
clear all
close all

% load 10% sample data and the test data
load DDISample.mat

% convert to normal format instead of sparse
Classp_train=full(Classp_train);
Classm_train=full(Classm_train);    
Classp_test=full(Classp_test);
Classm_test=full(Classm_test);


% Set random number to an initial seed
[r,c]=size(Classm_train);
s=RandStream('mt19937ar','Seed',550);
%generate a permutation of the data
p=randperm(s,r);
Classm_train=Classm_train(p,:);
%featurenames=featurenames(p);




Classm_train = Classm_train(1:380,:);

%% PCA on data
Train_total = [Classp_train; Classm_train];

[mp,np] = size(Classp_train);    % size for Classp
[mm,nm] = size(Classm_train);    % size for Classm
[m,n] = size(Train_total);      % size for total

train_mean = (1/m)*(ones(1,m)*Train_total);

Train_total = Train_total - ones(m,1)*train_mean;


%%

[eigenvectors, scores, eigenvalues] = pca(Train_total);

%% 
trimmed_scores = scores(:,1:300);
classp_scores = trimmed_scores(1:mp,:);
classm_scores = trimmed_scores(mp+1:m,:);

%% Fisher


meanp=mean(classp_scores);
meanm=mean(classm_scores);

psize=size(classp_scores,1)
nsize=size(classm_scores,1)
Bp=classp_scores-ones(psize,1)*meanp;
Bn=classm_scores-ones(nsize,1)*meanm;

Sw=Bp'*Bp+Bn'*Bn;
wfisher = Sw\(meanp-meanm)';
wfisher=wfisher/norm(wfisher);

tfisher=(meanp+meanm)./2*wfisher
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Analyze training data  results of the Fisher Linear Discriminant

FisherPosErrorTrain = sum(classp_scores*wfisher <= tfisher);
FisherNegErrorTrain = sum(classm_scores*wfisher >= tfisher);

FisherTrainError= ((FisherPosErrorTrain + FisherNegErrorTrain)/(size(trimmed_scores,1)))  

% Histogram of Fisher Training Results
HistClass(classp_scores,classm_scores,wfisher,tfisher,...
    'Fisher Method Training Results',FisherTrainError); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%{
FisherPosErrorTest = sum(Classp_test*wfisher <= tfisher);
FisherNegErrorTest = sum(Classm_test*wfisher >= tfisher);

FisherTestError= ((FisherPosErrorTest + FisherNegErrorTest)/(size(Test,1)))   

% Histogram of Fisher Testing Results
HistClass(Classp_test,Classm_test,wfisher,tfisher,...
    'Fisher Method Testing Results',FisherTestError); 

%}




%%

A = Train_total' * trimmed_scores * wfisher;
absA = abs(A);

words = cell(30,1);
word_values = zeros(30,1);

for i = 1:30
    [M,I] = max(absA);
    words(i,1) = featurenames(I);
    word_values(i,1) = A(I,1);
    absA(I,1) = 0;
end

%% Compute test scores

Classm_test_scores = Classm_test * eigenvectors;
Classp_test_scores = Classp_test * eigenvectors;


%{
FisherPosErrorTest = sum(Classp_test*wfisher <= tfisher);
FisherNegErrorTest = sum(Classm_test*wfisher >= tfisher);

FisherTestError= ((FisherPosErrorTest + FisherNegErrorTest)/(size(Test,1)))   

% Histogram of Fisher Testing Results
HistClass(Classp_test,Classm_test,wfisher,tfisher,...
    'Fisher Method Testing Results',FisherTestError); 

%}

%% Fisher on Test

meanp_test=mean(Classp_test_scores);
meanm_test=mean(Classm_test_scores);

psize_test=size(Classp_test_scores,1);
nsize_test=size(Classm_test_scores,1);
Bp_test=Classp_test_scores-ones(psize_test,1)*meanp_test;
Bn_test=Classm_test_scores-ones(nsize_test,1)*meanm_test;

Sw_test=Bp_test'*Bp_test+Bn_test'*Bn_test;
wfisher_test = Sw_test\(meanp_test-meanm_test)';
wfisher_test=wfisher_test/norm(wfisher_test);

tfisher_test=(meanp_test+meanm_test)./2*wfisher_test

% Analyze training data  results of the Fisher Linear Discriminant

FisherPosErrorTrain = sum(Classp_test_scores*wfisher_test <= tfisher_test);
FisherNegErrorTrain = sum(Classm_test_scores*wfisher_test >= tfisher_test);

FisherTrainError= ((FisherPosErrorTrain + FisherNegErrorTrain)/(size(trimmed_scores,1)))  

% Histogram of Fisher Training Results
HistClass(Classp_test_scores,Classm_test_scores,wfisher_test,tfisher_test,...
    'Fisher Method Testing Results',FisherTrainError); 

%RESULTS using 380 sentences from each class 5.39% training, 39.87% testing