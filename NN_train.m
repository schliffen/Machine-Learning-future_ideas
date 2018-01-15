%Handwritten Recognition Project
% I did the simple one just for zero, next time for other digits too
% text time testing to see the performance if it was qualified
% thinking about possible impovements
% thinking about the demo
% 
%
function [W, w] = NN_train(tr_data, cost, nV, eta, batch_size, beta, mu, MAXIT)
% ENTER YOUR NAME HERE:
%% preparing the data
% preprocessing the data
% determine which data to load

[tr_m, tr_n] = size(tr_data);
tr_data = [standardize(tr_data(:,1:end-1),0.5*ones(1,tr_n-1),0.5*ones(1,tr_n-1)) tr_data(:,end)];

%% initial weights   ......................................................
load weight1

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
    Mean_E = 0;
    err = zeros(MAXIT,1);
    for it = 1:MAXIT,    
        tr_T = randi(tr_m,batch_size,1);     
        x = tr_data(tr_T,1:tr_n-1);     
        T = zeros(10,size(tr_T,1)); 
        % design target
        for i=1:batch_size
         T(tr_data(tr_T(i),tr_n)+1,i)=1;    % training target, temporary for now
        end
        [O, Oh, V, Vh] = forwardPass(x,w,W); 
        %
        % reshaping
        Oh = reshape(Oh, 10*batch_size,1);
        O = reshape(O, 10*batch_size,1);
        T = reshape(T, 10*batch_size,1);
        if strcmp(cost, 'mse')
             E = 0.5*norm(T-O,2)^2 ;      % SUM SQUARED ERROR
             % MSE derivative case  
             dW = -reshape((T-O).*relder(Oh),10,batch_size)*V;   % SUM SQUARED ERROR BASED dW
             dw = -((reshape(((T-O).*relder(Oh)),batch_size,10)*W).*relder(Vh))'*x ;%+ 0.04*w; % SUM SQUARED ERROR BASED dw
             
         elseif strcmp(cost, 'cross')
%         % Cross validation objective function
%              E = -sum(T.*log10(O) + (1-T).*log10(1-O));
%              dW = -(1/(log2(10)))*((T.*sigder(Oh)./O)*V +((1-T).*sigder(Oh)./(1-O))*V);  % my Tuning
%              dw = -(1/(log2(10)))*(((T.*sigder(Oh)./O)'*W).*sigder(Vh))'*x  + (1/(log2(10)))*((((1-T).*sigder(Oh)./(1-O))'*W).*sigder(Vh))'*x; % My tuning
        end    
        Mean_E =  Mean_E + ((sqrt(E)/batch_size - Mean_E)/MAXIT);  
        err(it+1)= sqrt(Mean_E)/batch_size;    
        if (mod(it,10000)==0)    % show learnig progress 
            cla;
            plot(0:10000:it,err(1:10000:it+1));
            xlabel('iteration');
            ylabel('squared error');
            drawnow;                
        end;
%      
  if it==1
        vect_p1 = -eta*dW; 
  end;
      vect1 = mu*vect_p1 - eta*dW ; %
      W = W - mu * vect_p1 + (1+mu)*vect1;
      vect_p1 = vect1;
%     
 if it==1
     vect_p = -eta*dw; 
 end;
    % Accelerated Gradient 
        vect = mu*vect_p - eta*dw + beta*rand(nV,tr_n-1); %  
        w = w - mu * vect_p + (1+mu)*vect;  
        vect_p = vect; 
     
    end %iteration
   if (it > 9000) 
    plot((0:10000:length(err)-1),err(1:10000:end)); 
    xlabel('iteration');
    ylabel('squared error');
    drawnow;
   end
    ms_err = 0;
    ms_err = ms_err/tr_m;
    rms_err = ms_err^0.5;
    if mod(it,5000)==0 
       %fprintf('RMS error over data points:%3.5f\n',rms_err);
       title(sprintf('RMS error:%3.5f\n',rms_err));
       
    end
%%
% save 
save('weight1.mat', 'W','w');
end % end for trainer
%end  % main


