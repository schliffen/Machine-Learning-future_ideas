% activation function
function ret = sigmoid(x)
B = 1.8;
ret = 1./(1+exp(-B*x));
end
%
% x sigmoid
%function ret = sigmoid(x)
%B = 1;
%ret = x./(1+exp(-B*x));
%end

% leaky relu
% function ret = sigmoid(x)
% ret = x;
% alpha = 0.19;
% ret(find(x<0)) = alpha .* x(find(x<0));
% end