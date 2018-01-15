% Testing 
% Author Ali Nehrani

% there is teo steps: first we collect data and then  preint the results

% 
n_test = 100;
acprt=0;
if ~ exist('Tdata')
    Tdata = [];
end
dt=0.01;
fig=figure(1);
for j = 1:n_test
    y_true = input('Enter true y: ')
    [x,y] = getUserTraj(dt, fig);
    test_data = [x y];
    Tdata=[Tdata; x y y_true];
    [O,Oh,V,Vh] = forwardPass(test_data,w,W); 
    [o_m,O_i]=max(O);
    y_pred = O_i-1;
    fprintf('Network digit estimation is %d\n:', y_pred);
    
    if y_pred == y_true
      acprt = acprt + 1;  
    end
    fprintf('Testing accuracy equals to: %f\n',acprt*100/j );
end

