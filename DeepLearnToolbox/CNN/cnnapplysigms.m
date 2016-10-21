%% update alpha and beta of sigmoid function
function net = cnnapplysigms(net, opts)

for l = 2 : numel(net.layers)
    if strcmp(net.layers{l}.type, 'c')
        for j = 1 : numel(net.layers{l}.a)
          net.layers{l}.sigm{j}.alpha = net.layers{l}.sigm{j}.alpha - opts.gamma * net.layers{l}.sigm{j}.dAlpha;
          net.layers{l}.sigm{j}.beta  = net.layers{l}.sigm{j}.beta  - opts.gamma * net.layers{l}.sigm{j}.dBeta;
        end
    end
end

net.ffSigm.alpha = net.ffSigm.alpha - opts.gamma * net.ffSigm.dAlpha;
net.ffSigm.beta  = net.ffSigm.beta  - opts.gamma * net.ffSigm.dBeta;

end
