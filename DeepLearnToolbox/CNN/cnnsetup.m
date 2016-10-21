function net = cnnsetup(net, x, y)
    assert(~isOctave() || compare_versions(OCTAVE_VERSION, '3.8.0', '>='), ['Octave 3.8.0 or greater is required for CNNs as there is a bug in convolution in previous versions. See http://savannah.gnu.org/bugs/?39314. Your version is ' myOctaveVersion]);
    inputmaps = 1;                          % set to 1, specific for gray image
    mapsize = size(squeeze(x(:, :, 1)));    % size of feature map, getting smaller progressively
    fprintf('original feature map size: %d * %d \n', mapsize(1), mapsize(1));
    for l = 1 : numel(net.layers)   %  layer
        fprintf('layer %d \n', l);
        % subsample layer
        if strcmp(net.layers{l}.type, 's')
            fprintf('\t pooling layer \n');
            % net.layers{l}.scale = pooling size
            mapsize = mapsize / net.layers{l}.scale;
            fprintf('\t channel: %d \n', inputmaps);
            fprintf('\t feature map size: %d * %d \n', mapsize(1), mapsize(2));
            assert(all(floor(mapsize)==mapsize), ['Layer ' num2str(l) ' size must be integer. Actual: ' num2str(mapsize)]);
            % inputmap = feature maps of current layer
            for j = 1 : inputmaps
                net.layers{l}.b{j} = 0;
            end
        end
        
         % convolution layer
        if strcmp(net.layers{l}.type, 'c')
            fprintf('\t convolution layer \n');
            % compute size of feature map after convolution
            mapsize = mapsize - net.layers{l}.kernelsize + 1;
            fprintf('\t feature map size: %d * %d \n', mapsize(1), mapsize(2));
            % number of learnable parameters in this layer
            fan_out = net.layers{l}.outputmaps * net.layers{l}.kernelsize ^ 2;
            fprintf('\t channel: %d \n', net.layers{l}.outputmaps);
            fprintf('\t number of convolution kernels %d * %d \n', inputmaps, net.layers{l}.outputmaps);
            fan_in = inputmaps * net.layers{l}.kernelsize ^ 2;
            for j = 1 : net.layers{l}.outputmaps  %  output map
                for i = 1 : inputmaps  %  input map
                    net.layers{l}.k{i}{j} = (rand(net.layers{l}.kernelsize) - 0.5) * 2 * sqrt(6 / (fan_in + fan_out));
                end
                net.layers{l}.b{j} = 0;
            end
            % number of feature maps changes in convolution layer
            % but does not change in pooling layer
            % continue to next layer
            inputmaps = net.layers{l}.outputmaps;
            
           %% adjustable parameters of sigmoid
            for j = 1 : net.layers{l}.outputmaps
                net.layers{l}.sigm{j}.alpha = ones(mapsize(1), mapsize(2));
                net.layers{l}.sigm{j}.beta  = zeros(mapsize(1), mapsize(2));
                net.layers{l}.sigm{j}.dAlpha = zeros(mapsize(1), mapsize(2));
                net.layers{l}.sigm{j}.dBeta  = zeros(mapsize(1), mapsize(2));
            end          
        end
    end
    
    % 'onum' is the number of labels, that's why it is calculated using size(y, 1). If you have 20 labels so the output of the network will be 20 neurons.
    % 'fvnum' is the number of output neurons at the last layer, the layer just before the output layer.
    % 'ffb' is the biases of the output neurons.
    % 'ffW' is the weights between the last layer and the output neurons. Note that the last layer is fully connected to the output layer, that's why the size of the weights is (onum * fvnum)
    fvnum = prod(mapsize) * inputmaps;
    onum = size(y, 1);
    fprintf('fc layer \n');
    fprintf('\t feature vector: %d \n', fvnum);
    fprintf('\t weight: %d * %d \n', onum, fvnum);
    fprintf('layer output %d \n', onum);
    % fully-connected layer at the last
    % initialize weight and bias
    net.ffb = zeros(onum, 1);
    net.ffW = (rand(onum, fvnum) - 0.5) * 2 * sqrt(6 / (onum + fvnum));
    
    %% adjustable parameters of sigmoid
    net.ffSigm.alpha = ones(onum, 1);
    net.ffSigm.beta = zeros(onum, 1);
    net.ffSigm.dAlpha = zeros(onum, 1);
    net.ffSigm.dBeta = zeros(onum, 1);
 end
