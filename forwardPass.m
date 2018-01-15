% forward pass of neural network
function [O,Oh,V,Vh] = forwardPass(x,w,W) 
Vh = x*w';
V = lrelu(Vh);
    %V(end) = 1;
    %V(end-1) = 0;
Oh = W*V';
O = lrelu(Oh);
z= []; 
for i = 1:size(O,1)
    z = [z; sum(exp(O))];
end
O = exp(O)./z;
end