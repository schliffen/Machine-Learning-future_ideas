function [SData, mean_X, std_X] = standardize(varargin)
switch nargin
    case 1
        mean_X = mean(varargin{1});
        std_X = std(varargin{1});

        SData = varargin{1} - repmat(mean_X, [size(varargin{1}, 1) 1]);

        for i = 1:size(SData, 2)
            SData(:, i) =  SData(:, i) / std(SData(:, i));
        end     
    case 3
        mean_X = varargin{2};
        std_X = varargin{3};
        SData = varargin{1} - repmat(mean_X, [size(varargin{1}, 1) 1]);
        for i = 1:size(SData, 2)
            SData(:, i) =  SData(:, i) / std_X(:, i);
        end 
end