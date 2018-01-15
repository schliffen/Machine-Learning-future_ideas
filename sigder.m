% derivative of sigmoid function
function ret = sigder(x)
B = 1.8;
ret = B*sigmoid(x).*(1-sigmoid(x));
end

% x sig derivative
%function ret = sigder(x)
%B = 1;
%ret = x.*(B*sigmoid(x).*(1-sigmoid(x))) + 1./(1+exp(-B*x));
%end

% leaky relu der
% function ret = sigder(x)
% ret = ones(size(x));
% alpha = 0.19;
% ret(find(x<0)) = alpha;
% end