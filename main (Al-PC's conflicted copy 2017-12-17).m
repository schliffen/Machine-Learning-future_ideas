%
% main 
%
clear all 
close all
clc

% load data is done in the training function
terMAX = 10      % number of selecting training data set
MAXIT = 10000000;    % maximum iteration of training 
nV = 10;           % number of hidden units
eta =.0137;         % learning rate

% loading and preparing data
 tr_tg=[];
 for i=1:10
     tr_data = load_data(i-1); % first place is for zero and last place for 9
     tr_tg =[tr_tg; tr_data ones(size(tr_data,1),1)*(i-1)];      
 end

% training network
[W,w] = NN_train(tr_tg, nV, eta, terMAX,  MAXIT);

% getting test data
% we can set dt
% dt:  period of data to be returned (e.g. 0.01)
%dt = .011
fig = figure(2)
%
%
n_test = input('Enter the number of tests: ');
%
dt=0.01;
%
for j = 1:n_test
    [x,y] = getUserTraj(dt, fig);
    test_data = [x y];
    [O,Oh,V,Vh] = forwardPass(test_data,w,W); 
    [o_m,O_i]=max(O);
    y_pred = O_i-1;
    fprintf('Network digit estimation is %d\n:', y_pred);
end


% provide to the network
% get network weights
% get test samples
% test by network weights
% do report





% testing scenario 
