function simpleBP2_hw
% ENTER YOUR NAME HERE:
% for 5 hidden layer 
%%
clear all
clc
nX = 3;        % input dimension (do not change)
nV = 5;        % TUNE  (number of hidden units)
eta =.87;     % TUNE  (learning rate)

x = zeros(1,nX);  % input layer
V = zeros(1,nV);  % hidden layer
O = 0;            % output layer
%%
W = (rand(1,nV)-0.5)*0.05;   % hidden->output weights
w = (rand(nV,nX)-0.5)*0.01;  % input->hidden weights

% Do not change the training data
tr_data = [ -2    -2  1 1;
             2    -2  1 1;
            -2     2  1 1;
             2     2  1 1;
            -1    -1  1 0;
             1    -1  1 0;
            -1     1  1 0;
             1     1  1 0;
            -0.2 -0.2 1 1;
               2 -0.2 1 1;
            -0.2  0.2 1 1;
             0.5  0.2 1 1
               2  0.0 1 0
              -2  0.0 1 0];
tr_x = tr_data(:,1:nX);
tr_T = tr_data(:,end);

[tr_m, tr_n] = size(tr_data);
figure(1); clf;

MAXIT = 100000;    % TUNE 
% MAXIT = 50000;    % TUNE 
err = zeros(MAXIT,1);
%%
dw_old = (rand(nV,3)-0.5)
mu = .087
%%
for it = 1:MAXIT,    
    
    k = floor(rand*tr_m+1);
    x = tr_x(k,:);
    T = tr_T(k);
    [O, Oh, V, Vh] = forwardPass(x,w,W); 
 
    E = 0.5*(T-O)^2;         % SUM SQUARED ERROR
%    E = -T.*log10(O) - (1-T).*log10(1-O); % SUM CROSS ENTROPY
    err(it+1)=E;
    
    if (mod(it,1000)==0)   % show learnig progress 
        cla;
        plot(0:it,err(1:it+1));
        xlabel('iteration');
        ylabel('squared error');
        drawnow;                
    end;
    % ------MSE
    dW = -(T-O)*sigder(Oh)*V;   % SUM SQUARED ERROR BASED dW

    
    % ---------- MSE
    dw = -(T-O)*sigder(Oh)*(W.*sigder(Vh))'*x; % SUM SQUARED ERROR BASED dw
 
    
    W = W - eta*dW;
%    w = w - eta* dw+0.005*(rand(nV,3)-0.5);   % TUNE
%    grw = beta*dw_old + dw
%    dw1 = alpha*dw - mtm*dw
%     
%%
    if it==1
       vect_p = -eta*dw;
    end
    
    %w - sqmul*grw; %- 0.01*dw_old; %0.0005*(rand(nV,3)-0.5);   % TUNE
    
    % Accelerated Gradient 
    vect = mu*vect_p - eta*dw ; % 
    w = w - mu * vect_p + (1+mu)*vect; % 
    vect_p = vect;
    
%    dw_old = dw;
    if (mod(it,5000) == 0)  
     if E >= 1e-3
%        vect_p = -eta*dw;
        eta = eta + .001;
        mu =  mu + .001;
        W = (rand(1,nV)-0.5);   % hidden->output weights
        w = (rand(nV,nX)-0.5);  % input->hidden weights
    elseif (eta > .1 && mu >.1)
        eta = eta - 0.01;
        mu = mu - .01;
    end
    end
      
   
end
%%
plot((0:length(err)-1),err); 
xlabel('iteration');
ylabel('squared error');
drawnow;

ms_err = 0;
for k=1:tr_m,
    x = tr_x(k,:);
    T = tr_T(k); 
    [O, Oh, V, Vh] = forwardPass(x,w,W);
    ms_err = ms_err + (T-O)^2;
    
    fprintf('Desired Output %2.2f  == %2.2f (Network Ouput)\n',T,O);
end
ms_err = ms_err/tr_m;
rms_err = ms_err^0.5;
fprintf('RMS error over data points:%3.5f\n',rms_err);
title(sprintf('RMS error:%3.5f\n',rms_err));

%% ENTER YOUR code to show decision boundary here
% -----------------------------------------------
xmin = -2; %min(tr_x(:,1)); 
xmax = 2;%max(tr_x(:,1));
ymin = -2; %min(tr_x(:,2)); 
ymax = 2; %max(tr_x(:,2));
figure;
hold on
dx=0.1;
dy=0.1;

for x1=xmin:dx:xmax
	for y1=ymin:dy:ymax
        activation = forwardPass([x1 y1 1],w,W);
           if activation > 0.5
			plot(x1,y1,'*c', 'markersize', 9)
            else
			plot(x1,y1,'>m', 'markersize', 9)
           end
            
		hold on;
    end
end
legend('Class 1', 'Class 0');

 nn_outpt = zeros(size(tr_x,1),1);
 for i=1:size(tr_x,1)
     nn_outpt(i) = forwardPass(tr_x(i,:),w,W);
 end

%class = zeros(n_pairs, size(x,2)+size(O,2))
class1 = tr_x(find(nn_outpt>=.5),1:2);%(1:7,1:2)
class0 = tr_x(find(nn_outpt<.5),1:2);%(8:14,1:2)
plot(class1(:,1),class1(:,2), 'b>');
plot(class0(:,1),class0(:,2), 'go');
% include legend
legend('Class 1', 'Class 0');
%legend(); 
% label the axes.
xlabel('x');
ylabel('T');
% -----------------------------------------------

end  % main
%%


function [O,Oh,V,Vh] = forwardPass(x,w,W) 
Vh = x*w';
V = sigmoid(Vh);
    V(end) = 1;
    %V(end-1) = 0;
Oh = V*W';
O = sigmoid(Oh);
end

function ret = sigmoid(x)
B = 2;
ret = 1./(1+exp(-B*x));
end

function ret = sigder(x)
B = 2;
ret = B*sigmoid(x).*(1-sigmoid(x));
end


