%
% main  This  program do cross validation
%
clear all 
close all
clc

%% Define parameters
MAXIT = 200000;     % maximum iteration of training 
nV = 25;             % number of hidden units    
batch_size = 10;  % batch size
beta = 1e-10;  % randomizer
mu = 0.0001;  % optimization parameter
eta =0.0001;     % learning rate
fold = 10;   % number of folds
%% Cross Validation data splitting
%try 
%    data = [struct2cell(load('Data09.mat')){:}]; % IMPORTANT please Uncomment this when using OCTAVE
%catch
    tr_tg = load('Data09.mat');            % IMPORTANT please Uncomment this when using MATLAB
    data = struct2array(tr_tg); clear tr_tg % IMPORTANT please Uncomment this when using MATLAB
%end
%  
%
%load weight1.mat  % In the case of testing ( This loads Trained weights)
%%
[m_data, n_data] = size(data);
% ------------------------------------    using 10-fold cross validation --------------------
total = linspace(1,m_data,m_data);
test_sel = total;

Train_data_ad = {}; Test_data_ad = {}; 
% number of folds
test_num = floor(m_data/fold);
% Cross validation spliting data - the address of data stored only
for n_fold = 1:fold 
    T_sample = randi(size(test_sel,2),test_num,1); % selecting 10% of the data randomly
    Test_data_ad{n_fold} = T_sample;
    Train_data_ad{n_fold} = setdiff(total,T_sample);
    test_sel = setdiff(test_sel,T_sample);  
end
clear  T_sample test_num
%% Iteration on the folds
E_fold = [];
for n_fold = 1:fold
%% training process
% select type of the cost function:
%cost = 'cross'
cost = 'mse'
%cost = input('which kind of cost function do you prefer: enter "cross" for cross entropy and "mse" for mean square error: ') 
[W,w] = NN_train(data(Train_data_ad{n_fold},:), cost, nV, eta, batch_size, beta, mu, MAXIT);

%% Testing Process 
    n_true = 0;
    test_data = data(Test_data_ad{n_fold},:);  % data is loaded from test fold
    for i=1:size(test_data,1)
       [O,Oh,V,Vh] = forwardPass(test_data(i,1:end-1),w,W); 
       [o_m,O_i]=max(O);
       y_pred = O_i-1;
       if y_pred == test_data(i,end)
          n_true = n_true + 1;
       end
    end
    E_fold = [E_fold n_true/size(test_data,1)*100];
    fprintf('Test accuracy in fold %d is equal to: %f percent \n', n_fold, n_true/size(test_data,1)*100);
    
 % clculating training accuracy   
 train_data = data(Train_data_ad{n_fold},:);
 E_train = [];
 n_true = 0;
    for i=1:size(train_data,1)
       [O,Oh,V,Vh] = forwardPass(train_data(i,1:end-1),w,W); 
       [o_m,O_i]=max(O);
       y_pred = O_i-1;
       if y_pred == train_data(i,end)
          n_true = n_true + 1;
       end
    end
    E_train = [E_train n_true/size(train_data,1)*100];
    fprintf('Train accuracy in fold %d is equal to: %f percent \n', n_fold, n_true/size(train_data,1)*100);
clear W w  test_data   
end
fprintf('Average Test accuracy in %d folds is: %f \n', fold, mean(E_fold));
fprintf('Average Train accuracy in %d folds is: %f \n', fold, mean(E_train));

