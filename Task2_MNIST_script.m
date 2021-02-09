%Task:  Identifying subjects in the MNIST.mat dataset -- test and find
%accuracy of more than 90%

clear all
close all
data = load('mnist.mat');
imgs_train = data.imgs_train;
imgs_test = data.imgs_test;
labels_train = data.labels_train;
labels_test = data.labels_test;
%
[d1 d2 numTrain] = size(imgs_train);
Xtrain = zeros(d1,d2,1,numTrain);
for j = 1:numTrain
    Xtrain(:,:,1,j) = imgs_train(:,:,j);
end
[d1 d2 numTest] = size(imgs_test);
Xtest = zeros(d1,d2, 1, numTest);
for j = 1:numTest
    Xtest(:,:,1,j) = imgs_test(:,:,j);
end

figure
colormap gray
perm = randperm(numTest,20);
for i = 1:20
    subplot(4,5,i);
    imagesc(imgs_test(:,:,perm(i)));
end


%third test
layers = [...
    imageInputLayer([d1 d2 1])
    convolution2dLayer(7,30,'Padding',1)
    reluLayer
    batchNormalizationLayer
    maxPooling2dLayer(2,'Stride',2)
%    groupedConvolution2dLayer(3,24,2,'Padding','same')
%    reluLayer
%    crossChannelNormalizationLayer(5)
%    maxPooling2dLayer(2,'Stride',2)
%    convolution2dLayer(3,32,'Padding',1)
%    reluLayer
%    groupedConvolution2dLayer(3,36,1,'Padding','same')
    reluLayer
    groupedConvolution2dLayer(3,36,1,'Padding','same')
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
%    fullyConnectedLayer(512)
%    reluLayer
%    dropoutLayer(0.5)
    fullyConnectedLayer(512)
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];


%second test
%layers = [
%    imageInputLayer([d1 d2 1])  
%    convolution2dLayer(7,30,'Padding',1)
%    batchNormalizationLayer
%    reluLayer    
%    maxPooling2dLayer(2,'Stride',2) 
%    convolution2dLayer(3,32,'Padding',1)
%    batchNormalizationLayer
%    reluLayer 
%    fullyConnectedLayer(10)
%    softmaxLayer
%    classificationLayer];


%first test
%layers = [ ...
%    imageInputLayer([d1 d2 1])
%    convolution2dLayer(3,16)
%    reluLayer
%    maxPooling2dLayer(2,'Stride',2)
%    fullyConnectedLayer(10)
%    softmaxLayer
%    classificationLayer];

options = trainingOptions('sgdm', ...
    'MaxEpochs',10,...
    'InitialLearnRate',1e-3, ...
    'Verbose',false,...
    'Plots','training-progress');

net = trainNetwork(Xtrain,labels_train,layers,options);


labels_test_predicted = classify(net,Xtest);
accuracy = sum(labels_test_predicted == labels_test)/numel(labels_test)
