%Handwritten Recognition Project
% I did the simple one just for zero, next time for other digits too
% text time testing to see the performance if it was qualified
% thinking about possible impovements
% thinking about the demo
% 
%

function [W, w] = NN_train(tr_data, nV, eta, terMAX,  MAXIT)
% ENTER YOUR NAME HERE:

%% six neuron case  .......................................................
%nX = 3;        % input dimension (do not change)
%nV = 6;        % TUNE  (number of hidden units)
%eta =.87;      % TUNE  (learning rate)

%% preparing the data
% preprocessing the data
% determine which data to load

%%
%for trainer = 1:terMAX
    [tr_m, tr_n] = size(tr_data);
%tr_T = tr_data(:,101:200);

% tr_x=[]; tr_T=[];
% for i=1:size(P_tr_x,1)
%    tr_x = [tr_x P_tr_x(i,:)];
%    tr_T = [tr_T P_tr_T(i,:)];
% end

%% initial weights   ......................................................
%tr_x = tr_x'; tr_T = tr_T';
%    if trainer ==1
    if ~ (exist('W') && exist('w'))
        W = (rand(10,nV)-0.5)*0.05;   % hidden->output weights
        w = (rand(nV,tr_n-1)-0.5)*0.01;  % input->hidden weights
    end;

%% initial parameters   ...................................................
    x = zeros(1,tr_n-1);   % input layer
    V = zeros(nV,tr_n-1);  % hidden layer
%    O = 0;            % output layer

%% setting the parameters 
    figure(1); clf;
    err = zeros(MAXIT,1);
%    dw_old = (rand(nV,tr_n)-0.5);
%    E_old =1;
    beta = 0.0008;
    mu = .017;
    
    for it = 1:MAXIT,    
        tr_T= randi(tr_m,1); % testing with just 1
         %tr_data = load_data(tr_T-1); % first place is for zero and last place for 9
        x = tr_data(tr_T,1:tr_n-1);
        T = zeros(10,1); T(tr_data(tr_T,tr_n)+1)=1;    % training target, temporary for now
        %k = floor(rand*tr_m+1);
        %x = tr_x(k,:);    % training input
        [O, Oh, V, Vh] = forwardPass(x,w,W); 
        E = 0.5*norm(T-O,2)^2;         % SUM SQUARED ERROR
        err(it+1)=E;    
        if (mod(it,1000)==0)   % show learnig progress 
            cla;
            plot(0:it,err(1:it+1));
            xlabel('iteration');
            ylabel('squared error');
            drawnow;                
        end;
   
        dW = -((T-O).*sigder(Oh))*V;   % SUM SQUARED ERROR BASED dW
        dw = -((((T-O).*sigder(Oh))'*W).*sigder(Vh))'*x; % SUM SQUARED ERROR BASED dw
    
        W = W - eta*dW;
%    w = w - eta* dw+0.005*(rand(nV,3)-0.5);   % TUNE
%    grw = beta*dw_old + dw
%    dw1 = alpha*dw - mtm*dw
%     
        if it==1
            vect_p = -eta*dw; 
        end;
    
    %w - sqmul*grw; %- 0.01*dw_old; %0.0005*(rand(nV,3)-0.5);   % TUNE
    
    % Accelerated Gradient 
        vect = mu*vect_p - eta*dw + beta*rand(nV,tr_n-1); %
    
        w = w - mu * vect_p + (1+mu)*vect;
    
        vect_p = vect;
    
        if E < 1e-8
            beta = beta*.0001;
        end;
%    dw_old = dw;

%    if (mod(it,MAXIT/50) == 0)
%    if E >= E_old
%        eta = .02*eta
%    else
%        eta = .01*eta;     
    end

    plot((0:length(err)-1),err); 
    xlabel('iteration');
    ylabel('squared error');
    drawnow;
    ms_err = 0;
%      for k=1:tr_m,
%          x = tr_x(k,:);
%          [O, Oh, V, Vh] = forwardPass(x,w,W);
%          ms_err = ms_err + norm(T-O,2);%^2;
%          [o_m,O_i]=max(O);
%          [o_m,T_i]=max(T);
%          fprintf('Desired Output %2.2f  == %2.2f (Network Ouput)\n',T_i-1,O_i-1);
%      end;

    ms_err = ms_err/tr_m;
    rms_err = ms_err^0.5;
    fprintf('RMS error over data points:%3.5f\n',rms_err);
    title(sprintf('RMS error:%3.5f\n',rms_err));

end % end for trainer

%end  % main


