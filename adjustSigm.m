function [Y] = adjustSigm(X, net, l, j)
% X     activation
% net   cnn structure
% l     layer
% j     map

n = numel(net.layers);
if l == n 
    U = repmat(net.ffSigm.alpha, [1 size(X, 2)]) .* X ...
        + repmat(net.ffSigm.beta, [1 size(X, 2)]);
%     Y = 1 ./ (1 +exp(-U));
else
    U = repmat(net.layers{l}.sigm{j}.alpha, [1 1 size(X, 3)]) .* X ...
        + repmat(net.layers{l}.sigm{j}.beta, [1 1 size(X, 3)]);
%     Y = 1 ./ (1 + exp(-U));
end

Y = 1 ./ (1 + exp(-U)); 

end

