function learntModel = learn(numberOfSamples)
    pkg load image nnet
    if nargin == 1 && isnumeric(numberOfSamples) && numberOfSamples > 0
        inputSamples = {};
        outputSamples = {}
        for i = 1:numberOfSamples
            printf("learning from sample %d\n", i)
            samplePic = imread(["samples/cotton/" num2str(i) "_input.jpg"]);
            nnInputSize = [100,133];
            samplePicSize = size(samplePic);
            nnInputAspect = nnInputSize(2)/nnInputSize(1);
            rawSampleAspect = samplePicSize(2)/samplePicSize(1);
            if (rawSampleAspect < nnInputAspect) 
                #scale to match height
                resizedPic = imresize(samplePic, [ nnInputSize(1) NaN ]);
                #pad to match width
                prePaddedPic = padarray(resizedPic, [0,floor((nnInputSize(2) - size(resizedPic)(2))/2)], 0, 'pre');
                paddedPic = padarray(prePaddedPic, [0,ceil((nnInputSize(2) - size(resizedPic)(2))/2)], 0, 'post');
            elseif (rawSampleAspect > nnInputAspect)
                #scale to match width
                resizedPic = imresize(samplePic, [ NaN nnInputSize(2) ]);
                #pad to match height
                prePaddedPic = padarray(resizedPic, [floor((nnInputSize(1) - size(resizedPic)(1))/2),0], 0, 'pre');
                paddedPic = padarray(prePaddedPic, [ceil((nnInputSize(1) - size(resizedPic)(1))/2),0], 0, 'post');
            else
                paddedPic = imresize(samplePic, [nnInputSize(1) NaN]);
            endif
            #size(samplePic)
            #size(paddedPic)
            #imshow(paddedPic);
            #figure;
            inputSamples{i} = getVectorFromImage(paddedPic);
            outputPic = edge(rgb2gray(paddedPic), 'Canny');
            #imshowpair(paddedPic, outputPic, 'montage');
            #imshow(outputPic);
            #figure;
            outputSamples{i} = getVectorFromImage(outputPic);
        endfor
        inputSamples = cell2mat(inputSamples);
        size(inputSamples)
        outputSamples = cell2mat(outputSamples);
        size(outputSamples)
            
        #net = feedforwardnet([nnInputSize(1)*nnInputSize(2), nnInputSize(1)*nnInputSize(2)]);
        #net.layers{:}.transferFcn = 'logsig';
        inputMinMaxValues = [zeros(size(inputSamples(1), 1), 1) 255 * ones(size(inputSamples(1), 1), 1)];
        net = newff(inputMinMaxValues, [nnInputSize(1)*nnInputSize(2), size(outputSamples(1),1)]);
        net.inputs.size
        net.outputs.size
        #net = configure(net,inputSamples,outputSamples);
        net = train(net,inputSamples,outputSamples);
        #view(net);
        y = net(inputSamples);
        perf = perform(net,y,outputSamples)
    elseif
        usage("invalid usage of 'learn'. Try: learn(numberOfSamples)")
    endif
endfunction
function imageVector = getVectorFromImage(inputImage)
    [I, J, K] = size(inputImage);
    imageVector = zeros(I * J * K,1);
    for i = 1:I
        for j = 1:J
            for k = 1:K
                imageVector(((i-1)*J*K) + ((j-1)*K) + (k-1) + 1,1) ...
                = inputImage(i, j, k);
            endfor
        endfor
    endfor
endfunction
