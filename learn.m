function learntModel = learn(numberOfSamples)
import image.* nnet.*
if nargin == 1 && isnumeric(numberOfSamples) && numberOfSamples > 0
    inputSamples = {};
    outputSamples = zeros(2, numberOfSamples);
    for i = 1:numberOfSamples
        disp('learning from sample ' + i)
        samplePic = imread(['samples/zerosAndOnes/' num2str(i) '_input.jpg']);
        nnInputSize = [30, 30];
        samplePicSize = size(samplePic);
        nnInputAspect = nnInputSize(2)/nnInputSize(1);
        rawSampleAspect = samplePicSize(2)/samplePicSize(1);
        if (rawSampleAspect < nnInputAspect)
            % scale to match height
            resizedPic = imresize(samplePic, [ nnInputSize(1) NaN ]);
            % pad to match width
            sizeOfResizedPic = size(resizedPic);
            prePaddedPic = padarray(resizedPic, [0,floor((nnInputSize(2) - sizeOfResizedPic(2))/2)], 0, 'pre');
            paddedPic = padarray(prePaddedPic, [0,ceil((nnInputSize(2) - sizeOfResizedPic(2))/2)], 0, 'post');
        elseif (rawSampleAspect > nnInputAspect)
            % scale to match width
            resizedPic = imresize(samplePic, [ NaN nnInputSize(2) ]);
            % pad to match height
            sizeOfResizedPic = size(resizedPic);
            prePaddedPic = padarray(resizedPic, [floor((nnInputSize(1) - sizeOfResizedPic(1))/2),0], 0, 'pre');
            paddedPic = padarray(prePaddedPic, [ceil((nnInputSize(1) - sizeOfResizedPic(1))/2),0], 0, 'post');
        else
            paddedPic = imresize(samplePic, [nnInputSize(1) NaN]);
        end
        % size(samplePic)
        % size(paddedPic)
        % imshow(paddedPic);
        % figure;        
        inputSamples{i} = getVectorFromImage(rgb2gray(paddedPic));        
%         imshowpair(paddedPic, outputPic, 'montage');
%         imshow(outputPic);
%         figure;        
        outputSamples(mod(i+1,2) + 1,i) = 1;        
    end
    inputSamples = cell2mat(inputSamples);
    size(inputSamples)
    size(outputSamples)    
%     net = feedforwardnet([nnInputSize(1)*nnInputSize(2), nnInputSize(1)*nnInputSize(2)]);
%     net.layers{:}.transferFcn = 'logsig';
%     net = configure(net,inputSamples,outputSamples);
    
    net = newff(inputSamples, outputSamples, [1000 500 100 50 40 30 10] , {'tansig', 'tansig', 'tansig', 'tansig', 'tansig', 'tansig', 'tansig', 'logsig'},'trainrp');
    net = train(net,inputSamples,outputSamples);
    
%     view(net);
%     y = net(inputSamples);
%     perf = perform(net,y,outputSamples);
%     disp(perf);
    learntModel = net;
    j = 1;
    for inputSample = inputSamples
    testOutputVector = sim(learntModel, inputSample);
    displayVector = (testOutputVector == max(testOutputVector));
    disp([ num2str(j) '_input']);
    disp(displayVector);
    j = j + 1;
    end
else
    usage('invalid usage of "learn". Try: learn(numberOfSamples)')
end
end
function imageVector = getVectorFromImage(inputImage)
[I, J, K] = size(inputImage);
imageVector = zeros(I * J * K,1);
for i = 1:I
    for j = 1:J
        for k = 1:K
            imageVector(((i-1)*J*K) + ((j-1)*K) + (k-1) + 1,1) ...
                = inputImage(i, j, k);
        end
    end
end
end
function normalImage = getImageFromVector(imageVector, I, J, K)
normalImage = zeros(I, J, K);
for i = 1:I
    for j = 1:J
        for k = 1:K
            normalImage(i, j, k) = ...
            imageVector(((i-1)*J*K) + ((j-1)*K) + (k-1) + 1,1);
                
        end
    end
end
end