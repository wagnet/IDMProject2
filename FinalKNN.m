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

%%

Train = [Classp_train;Classm_train];
Test = [Classp_test;Classm_test];

[ptrain_m,ptrain_n]=size(Classp_train);
[mtrain_m,mtrain_n]=size(Classm_train);
[ptest_m,ptest_n]=size(Classp_test);
[mtest_m,mtest_n]=size(Classm_test);

YTrain = [ones(ptrain_m,1);-ones(mtrain_m,1)];
YTest = [ones(ptest_m,1);-ones(mtest_m,1)];

%%

train_mean = (1/ptrain_m+mtrain_m)*(ones(1,ptrain_m+mtrain_m)*Train);

%Train = Train - ones(ptrain_m+mtrain_m,1)*train_mean;
%Test = Test - ones(ptest_m+mtest_m,1)*train_mean;


%Train_total = Train_total - ones(m,1)*train_mean;

%%

classifier=knnsearch(Train,Test);
total_error=0;
[s,z]=size(Test)

%%
for i=1:s,
    if(YTest(i)~=YTrain(classifier(i)))
        total_error=total_error+1;
    end
end
error_percent = total_error/s