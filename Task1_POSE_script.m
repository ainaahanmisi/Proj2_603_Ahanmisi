%Task:  Identifying subjects in the pose.mat dataset -- test and find
%accuracy of more than 50%

clear all
close all
data = load('pose.mat');
pose = data.pose;
[d1, d2, numPose, numPeop] = size(pose);

numTrain = 10;
numTest = numPose - numTrain;
Xtrain = zeros(d1,d2,1,numPeop*numTrain);
for jnumPeop = 1:numPeop
    jstart = (jnumPeop-1)*numTrain;
    for jpos = 1:numTrain
        Xtrain(:,:,1,jstart+jpos)=pose(:,:,jpos,jnumPeop);
    end
end
Label_train = categorical((kron(1:numPeop,ones(1,numTrain)))');

%test set
Xtest = zeros(d1,d2,1,numPeop*numTest);
for jnumPeop = 1:numPeop
    jstart = (jnumPeop-1)*numTest;
    for jpos = 1:numTest
        Xtest(:,:,1,jstart+jpos)=pose(:,:,numTrain+jpos,jnumPeop);
    end
end
Label_test = categorical((kron(1:numPeop,ones(1,numTest)))');

%third test
layers = [ ...
    imageInputLayer([d1,d2,1])
    convolution2dLayer(5,16,'Padding',1)
    reluLayer
    batchNormalizationLayer
    maxPooling2dLayer(2,'Stride',2)
    groupedConvolution2dLayer(3,24,2,'Padding','same')
    reluLayer
    batchNormalizationLayer
%    maxPooling2dLayer(2,'Stride',2)
%    convolution2dLayer(3,32,'Padding',1)
%    reluLayer
%    groupedConvolution2dLayer(3,36,2,'Padding','same')
%    reluLayer
    maxPooling2dLayer(2,'Stride',2)
%    fullyConnectedLayer(512)
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(numPeop)
    softmaxLayer
    classificationLayer];

%second test
%layers = [
%    imageInputLayer([d1 d2 1])  
%    convolution2dLayer(7,20,'Padding',1)
%    batchNormalizationLayer
%    reluLayer    
%    maxPooling2dLayer(2,'Stride',2) 
%    convolution2dLayer(3,32,'Padding',1)
%    batchNormalizationLayer
%    reluLayer 
%    fullyConnectedLayer(numPeop)
%    softmaxLayer
%    classificationLayer];


%first test
%layers = [ ...
%    imageInputLayer([d1 d2 1])
%    convolution2dLayer(5,20)
%    reluLayer
%    maxPooling2dLayer(2,'Stride',2)
%    fullyConnectedLayer(numPeop)
%    softmaxLayer
%    classificationLayer];

options = trainingOptions('sgdm', ...
    'MaxEpochs',120,...
    'InitialLearnRate',1e-2, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(Xtrain,Label_train,layers,options);


Label_test_predicted = classify(net,Xtest);
accuracy = sum(Label_test_predicted == Label_test)/numel(Label_test)
