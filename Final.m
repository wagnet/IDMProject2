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

%% PCA on data

Train_total = [Classp_train; Classm_train];



[m,n] = size(Train_total);
train_mean = (1/m)*(ones(1,m)*Train_total);

Train_total = Train_total - ones(m,1)*train_mean;


%%

[eigenvectors, scores, eigenvalues] = pca(Train_total);

