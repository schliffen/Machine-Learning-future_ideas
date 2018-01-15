function ret = relder(x)
ret = ones(size(x));
alpha = 0.19;
ret(find(x<0)) = alpha;
end