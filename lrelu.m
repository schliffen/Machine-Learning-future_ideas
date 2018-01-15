function ret = lrelu(x)
ret = x;
alpha = 0.19;
ret(find(x<0)) = alpha .* x(find(x<0));
end