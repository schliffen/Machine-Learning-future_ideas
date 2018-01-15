% creating data set

dt=0.01
fig = figure(1)

if ~ exist('DataN'); 
   DataN = [];
end

n =9 %= input('enter the target you want to collect data: ');

for i=1:189
    [x,y] = getUserTraj(dt,fig);
    DataN = [DataN; [x y] n]; 
%    y = input('Do you want to continue?');
%    if y == 'n'
%        break
%    end
end
%data2 = [load_data(0) zeros(13,1)];
