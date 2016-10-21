% net:  cnn structure
% y:    batch labels, class vector * batchNumber
function net = cnnbp(net, y)
n = numel(net.layers);
% error = label - output
net.e = y - net.o;
%  loss function
net.L = 0.5 * sum(net.e(:) .^ 2) / size(net.e, 2);

%%  backprop deltas
%     % delta of output layer
%     net.od = - net.e .* (net.o .* (1 - net.o)); % output delta
%% adjustable backprop deltas
net.od = - net.e .* net.o .* (1 - net.o) ...
    .* repmat(net.ffSigm.alpha, [1 size(net.o, 2)]);
%%
net.fvd = (net.ffW' * net.od);              % feature vector delta
% gradient of this layer depends on convolution layer or downsample
% layer, in downsampling layer, activation function is linear
if strcmp(net.layers{n}.type, 'c')      %  only execute, if conv layers is the last
    net.fvd = net.fvd .* (net.fv .* (1 - net.fv));
end

%  reshape feature vector deltas into output map style
%  mapsize * mapsize * batchNumber
sa = size(net.layers{n}.a{1});
fvnum = sa(1) * sa(2);
for j = 1 : numel(net.layers{n}.a)
    net.layers{n}.d{j} = reshape(net.fvd(((j - 1) * fvnum + 1) : j * fvnum, :), ...
        sa(1), sa(2), sa(3));
end

for l = (n - 1) : -1 : 1
    if strcmp(net.layers{l}.type, 'c')
        for j = 1 : numel(net.layers{l}.a)  
%                 % sigmoid function in convolution layer
%                 net.layers{l}.d{j} = net.layers{l}.a{j} .* (1 - net.layers{l}.a{j}) ...
            %% adjustable sigmoid function in convolution layer
            net.layers{l}.d{j} = repmat(net.layers{l}.sigm{j}.alpha, [1 1 size(net.layers{l}.a{j}, 3)]) ...
                .* net.layers{l}.a{j} .* (1 - net.layers{l}.a{j}) ...
                .* (expand(net.layers{l + 1}.d{j}, ...
                [net.layers{l + 1}.scale net.layers{l + 1}.scale 1]) ...
                / net.layers{l + 1}.scale ^ 2); % delta is divided by number of elemetns
        end
    elseif strcmp(net.layers{l}.type, 's')
        for i = 1 : numel(net.layers{l}.a)
            z = zeros(size(net.layers{l}.a{1}));
            for j = 1 : numel(net.layers{l + 1}.a)  
                % linear function in pooling layer
                 z = z + convn(net.layers{l + 1}.d{j}, ...
                     rot180(net.layers{l + 1}.k{i}{j}), 'full');
            end
            net.layers{l}.d{i} = z;
        end
    end
end

%%  calculate gradients
% gradient of delta with respect to convolution kernel and bias
% flipall is equivalent to rot180
for l = 2 : n
    if strcmp(net.layers{l}.type, 'c')
        for j = 1 : numel(net.layers{l}.a)
            for i = 1 : numel(net.layers{l - 1}.a)
                net.layers{l}.dk{i}{j} = convn(flipall(net.layers{l - 1}.a{i}), net.layers{l}.d{j}, 'valid') ...
                    / size(net.layers{l}.d{j}, 3);
            end
            net.layers{l}.db{j} = sum(net.layers{l}.d{j}(:)) / size(net.layers{l}.d{j}, 3);
           %% adjustable gradients of sigmoid function
            gradient = net.layers{l}.d{j} ./ repmat(net.layers{l}.sigm{j}.alpha, [1 1 size(net.layers{l}.d{j}, 3)]);
            gradAlpha = gradient .* net.layers{l}.z{j};
            gradBeta  = gradient;
            net.layers{l}.sigm{j}.dAlpha = mean(gradAlpha, 3);
            net.layers{l}.sigm{j}.dBeta  = mean(gradBeta, 3);                     

        end
    elseif strcmp(net.layers{l}.type, 's')
        for j = 1 : numel(net.layers{l}.a)
            net.layers{l}.db{j} = sum(net.layers{l}.d{j}(:)) / size(net.layers{l}.d{j}, 3);
        end            
    end
end

net.dffW = net.od * (net.fv)' / size(net.od, 2);
net.dffb = mean(net.od, 2);
%% adjustable gradients of sigmoid function
net.ffSigm.dAlpha = mean(- net.e .* net.o .* (1 - net.o) .* net.i, 2);
net.ffSigm.dBeta = mean(- net.e .* net.o .* (1 - net.o), 2);

%%
function X = rot180(X)
    X = flipdim(flipdim(X, 1), 2);
end

end
