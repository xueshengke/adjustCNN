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
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5)  % convolution layer
    struct('type', 's', 'scale', 2)                        % subsampling layer
    struct('type', 'c', 'outputmaps', 24, 'kernelsize', 5) % convolution layer
    struct('type', 's', 'scale', 2)                        % subsampling layer
};

opts.adjustable = 1;    % enable or disable adjustable function
opts.alpha = 0.1;         % learning rate
opts.gamma = 0.1;         % adjust rate
opts.scale = 0.985;     % annealing factor
opts.batchsize = 100;   % mini-batch for training through stochastic gradient descent
opts.numepochs = 1;    % training iteration

%% initialize convolutional neural network
fprintf('initialize convolutional neural network \n');
cnn = cnnsetup(cnn, trainData, trainLabel);

maxIteration = 100;
trainErrorRate = zeros(maxIteration, 1);
testErrorRate = zeros(maxIteration, 1);
for i = 1 : maxIteration
    %% train convolutional neural network
    fprintf('%d/%d, train convolutional neural network\n', i, maxIteration);
    [cnn, opts] = cnntrain(cnn, trainData, trainLabel, opts);
    trainErrorRate(i) = cnn.mse;
    
    %% test convolutional neural network
    fprintf('       test convolutional neural network\n');
    [ratio, error, bad] = cnntest(cnn, testData, testLabel);
    testErrorRate(i) = error;
    fprintf('accuracy: %.2f %% \n', ratio * 100 );
end

%% test convolutional neural network
[ratio, error, bad] = cnntest(cnn, testData, testLabel);
fprintf('final test accuracy: %.2f %% \n', ratio * 100 );

% plot mean squared error
figure;
plot(trainErrorRate); title('trainErrorRate');
figure;
plot(testErrorRate); title('testErrorRate');

% figure; plot(cnn.loss);

% assert(er<0.12, 'Too big error');
end