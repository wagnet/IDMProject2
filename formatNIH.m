%% Process NIH data 
% Note for debugging purposes readnih is limited to 10 lines. 
% Edit it to do the full run
maxlines=10000000000
%% read in postive test data
filename='TEST_DATA\features_test_pos_DB'
test_posmatrix=readnih(filename,maxlines);
% Check out size
size(test_posmatrix)
%Check how many columns are 0
sum(sum(test_posmatrix)>0)

%% Read in negative test data
filename='TEST_DATA\features_test_neg_DB'
test_negmatrix=readnih(filename,maxlines);
% Check out size
size(test_negmatrix)
%
sum(sum(test_negmatrix)>0)

%% Read in  netative train data
filename='TRAIN_DATA\features_train_neg_DB'
train_negmatrix=readnih(filename,maxlines);
% Check out size
size(train_negmatrix)
%
sum(sum(train_negmatrix)>0)
%% Read in  positive train data
filename='TRAIN_DATA\features_train_pos_DB'
train_posmatrix=readnih(filename,maxlines);
% Check out size
size(train_posmatrix)
%
sum(sum(train_posmatrix)>0)

%% save
save converted.mat train_posmatrix train_negmatrix test_posmatrix test_negmatrix
