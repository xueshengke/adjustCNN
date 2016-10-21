function mnistCNN
clear all;clc;
addpath(genpath('DeepLearnToolbox'));

%% load train data and test data
[trainData, trainLabel, testData, testLabel] = mnistGenerateData();
trainData = permute(trainData, [1 2 4 3]);
testData = permute(testData, [1 2 4 3]);

height = size(trainData, 1);
width = size(trainData, 2);
trainNumber = size(trainData, 3);
testNumber = size(trainData, 3);
classNumber = size(trainLabel, 1);

fprintf('prepare trainData %d * %d * %d \n', height, width, trainNumber);
fprintf('prepare trainLabel %d * %d \n', classNumber, trainNumber);
fprintf('prepare testData %d * %d * %d \n', height, width, testNumber);
fprintf('prepare testLabel %d * %d \n', classNumber, testNumber);

%% construct a convolutional neural network 
% after 100 epochs you'll get around 1.2% error
rand('state',0);
cnn.layers = {
    struct('type', 'i')                                    % input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5)  % convolution layer
    struct('type', 's', 'scale', 2)                        % subsampling layer
    struct('type', 'c', 'outputmaps', 16, 'kernelsize', 5) % convolution layer
    struct('type', 's', 'scale', 2)                        % subsampling layer
};

opts.alpha = 1;         % learning rate
opts.scale = 0.975;     % annealing factor
opts.batchsize = 100;   % mini-batch for training through stochastic gradient descent
opts.numepochs = 100;    % training iteration

%% initialize convolutional neural network
fprintf('initialize convolutional neural network \n');
cnn = cnnsetup(cnn, trainData, trainLabel);

%% train convolutional neural network
cnn = cnntrain(cnn, trainData, trainLabel, opts);

%% test convolutional neural network
[ratio, er, bad] = cnntest(cnn, testData, testLabel);
fprintf('accuracy: %.2f %% \n', ratio * 100 );

%plot mean squared error
figure; plot(cnn.loss);

% assert(er<0.12, 'Too big error');
end