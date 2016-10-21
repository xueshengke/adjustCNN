% net:  cnn structure
% x:    train data
% y:    train label
% opts: parameters

function [net, opts] = cnntrain(net, x, y, opts)
    % sample number
    num = size(x, 3);
    % number of batches
    numbatches = num / opts.batchsize;
    if rem(numbatches, 1) ~= 0
        error('numbatches not integer');
    end
    % save for value of object function
    net.loss = zeros(opts.numepochs * numbatches, 1);
    net.mse = 0;
    
    for i = 1 : opts.numepochs
        fprintf('epoch %d/%d, ', i, opts.numepochs);
        tic;
        
        % random sequence
        randIndex = randperm(num);
        
        for l = 1 : numbatches
            batch_x = x(:, :, randIndex((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            batch_y = y(:,    randIndex((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            
            % compute feedforward pass
            net = cnnff(net, batch_x);
            
            % compute back propagation
            % get gradients of error with respect to kernels and biases
            net = cnnbp(net, batch_y);
            
            % update W and b, through gradient descent method
            net = cnnapplygrads(net, opts);
            
            if opts.adjustable == 1
                net = cnnapplysigms(net, opts);
            end
            
            net.loss((i - 1) * numbatches + l) = net.L;
            
        end
        meanError = mean(net.loss((i - 1) * numbatches + 1 : i * numbatches));
        net.mse = meanError;
        fprintf('mean square error = %f \n', meanError);

        toc;
        fprintf('at present, opts.alpha=%f, opts.gamma=%f \n', opts.alpha, opts.gamma);
        opts.alpha = opts.alpha * opts.scale;
        opts.gamma = opts.gamma * opts.scale;
    end
    
end
