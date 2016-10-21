function [trainData, trainLabel, testData, testLabel] = mnistGenerateData()
%% load data in path
addpath(genpath('DeepLearnToolbox'));
load mnist_uint8;

imageSize = 28;
imageMaps = 1;
trainNumber = 60000;
testNumber = 10000;

train_x = double(train_x) ./ 255;
test_x = double(test_x) ./ 255;
train_y = double(train_y);
test_y = double(test_y);

%% normalize, mean to zero (optional)
% method 1: across each image
train_x = train_x - repmat(mean(train_x, 2), [1 size(train_x, 2)]);
test_x = test_x - repmat(mean(test_x, 2), [1 size(test_x, 2)]);

% % method 2: across all pixels
% train_x = train_x - repmat(mean(mean(train_x)), [size(train_x, 1) size(train_x, 2)]);
% test_x = test_x - repmat(mean(mean(test_x)), [size(test_x, 1) size(test_x, 2)]);

%% reconstruct data
trainData = reshape(train_x, trainNumber, imageMaps, imageSize, imageSize);
testData = reshape(test_x, testNumber, imageMaps, imageSize, imageSize);
trainData = permute(trainData, [4 3 2 1]);
testData = permute(testData, [4 3 2 1]);

trainLabel = train_y';
testLabel = test_y';
clear train_x train_y test_x test_y;

end
