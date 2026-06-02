%% MODEL 1: 2D-CNN (Visual) + 2D-CNN (Audio Mel-Spec) LATE FUSION
clear; clc; close all;

%% 1. YOLLAR VE PARAMETRELER
imgBaseDir = 'C:\Users\STB\Desktop\DATASET\CREMAD\Image'; 
audDir     = 'C:\Users\STB\Desktop\DATASET\CREMAD\Speech';      
saveDir    = 'C:\Users\STB\Desktop\Araştırma\Model1_2D_2D_Late';
if ~exist(saveDir, 'dir'), mkdir(saveDir); end

visImgSize = [224 224 3];  
melImgSize = [128 128 1];  % Ses bu sefer tekrar 2D Mel-Spektrogram resmine dönüştü
audLen = 48000; fs = 16000;
batchSize = 16; lr = 0.0001; maxEpochs = 40;

fprintf('\n🚀 MODEL 1: 2D-CNN (Vis) + 2D-CNN (Aud) LATE FUSION BAŞLIYOR...\n\n');

%% 2. SINIF BAZLI RASTGELE SPLIT (%80-%20)
classFolders = dir(imgBaseDir);
classFolders = classFolders([classFolders.isdir] & ~ismember({classFolders.name}, {'.', '..'}));
classNames = {classFolders.name};
numClasses = numel(classNames);

trainImg = {}; trainAud = {}; trainLbl = [];
testImg  = {}; testAud  = {}; testLbl  = [];

allAud = dir(fullfile(audDir, '**', '*.wav'));
audNames = {allAud.name};
audFolders = {allAud.folder};

for c = 1:numClasses
    cName = classNames{c};
    imgsInClass = dir(fullfile(imgBaseDir, cName, '*.jpg'));
    nFiles = numel(imgsInClass);
    
    idx = randperm(nFiles);
    nTrain = round(0.80 * nFiles);
    
    for i = 1:nFiles
        imgFile = fullfile(imgsInClass(idx(i)).folder, imgsInClass(idx(i)).name);
        
        parts = split(imgsInClass(idx(i)).name, '_frame');
        targetAudName = [parts{1}, '.wav'];
        matchIdx = find(strcmpi(audNames, targetAudName), 1);
        
        if ~isempty(matchIdx)
            audFile = fullfile(audFolders{matchIdx}, audNames{matchIdx});
            if i <= nTrain
                trainImg{end+1} = imgFile; trainAud{end+1} = audFile; trainLbl = [trainLbl; c];
            else
                testImg{end+1}  = imgFile; testAud{end+1}  = audFile; testLbl  = [testLbl; c];
            end
        end
    end
    fprintf('📁 Sınıf: %-8s -> %%80 Train, %%20 Test Ayrıldı.\n', cName);
end

YTrain = categorical(classNames(trainLbl)', classNames);
YTest  = categorical(classNames(testLbl)', classNames);

%% 3. DATASTORE VE MİNİBATCHQUEUE (İkisi de 2 Boyutlu Görüntü Formunda)
dsVisTrain = transform(arrayDatastore(trainImg', 'IterationDimension', 1), @(f) readRGBImage(f, visImgSize));
dsVisTest  = transform(arrayDatastore(testImg', 'IterationDimension', 1),  @(f) readRGBImage(f, visImgSize));

dsAudTrain = transform(arrayDatastore(trainAud', 'IterationDimension', 1), @(f) readMelSpecImage(f, audLen, fs, melImgSize));
dsAudTest  = transform(arrayDatastore(testAud', 'IterationDimension', 1),  @(f) readMelSpecImage(f, audLen, fs, melImgSize));

dsTrain = combine(dsVisTrain, dsAudTrain, arrayDatastore(YTrain));
dsTest  = combine(dsVisTest, dsAudTest, arrayDatastore(YTest));

% İki giriş de 2D CNN olacağı için Formatlar: 'SSCB', 'SSCB'
mbqTrain = minibatchqueue(dsTrain, 3, 'MiniBatchSize', batchSize, 'MiniBatchFcn', @(v,a,l) prepBatch(v,a,l,classNames), 'MiniBatchFormat', {'SSCB', 'SSCB', 'CB'});
mbqTest  = minibatchqueue(dsTest,  3, 'MiniBatchSize', batchSize, 'MiniBatchFcn', @(v,a,l) prepBatch(v,a,l,classNames), 'MiniBatchFormat', {'SSCB', 'SSCB', 'CB'});

%% 4. MİMARİ: LATE FUSION (2D-CNN & 2D-CNN)
lgraph = layerGraph();

% --- Görsel Dal (2D-CNN) ---
visBranch = [
    imageInputLayer(visImgSize, 'Name', 'in_vis', 'Normalization', 'zscore')
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'v_c1')
    reluLayer('Name', 'v_r1')
    maxPooling2dLayer(4, 'Stride', 4, 'Name', 'v_p1')
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'v_c2')
    reluLayer('Name', 'v_r2')
    globalAveragePooling2dLayer('Name', 'v_gap')
    % Geç Füzyon: Kendi özniteliğini özetliyor
    fullyConnectedLayer(128, 'Name', 'v_fc')
    reluLayer('Name', 'v_r3')
    dropoutLayer(0.5, 'Name', 'v_drop')
    flattenLayer('Name', 'v_flat')
]; 
lgraph = addLayers(lgraph, visBranch);

% --- İşitsel Dal (2D-CNN Mel-Spec) ---
audBranch = [
    imageInputLayer(melImgSize, 'Name', 'in_aud', 'Normalization', 'zscore')
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'a_c1')
    reluLayer('Name', 'a_r1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'a_p1')
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'a_c2')
    reluLayer('Name', 'a_r2')
    globalAveragePooling2dLayer('Name', 'a_gap')
    % Geç Füzyon: Kendi özniteliğini özetliyor
    fullyConnectedLayer(128, 'Name', 'a_fc')
    reluLayer('Name', 'a_r3')
    dropoutLayer(0.5, 'Name', 'a_drop')
    flattenLayer('Name', 'a_flat')
]; 
lgraph = addLayers(lgraph, audBranch);

% --- Geç Füzyon Birleştirme Masası ---
shared = [
    concatenationLayer(1, 2, 'Name', 'late_concat')
    fullyConnectedLayer(numClasses, 'Name', 'fc_out')
    softmaxLayer('Name', 'sm')
]; 
lgraph = addLayers(lgraph, shared);

lgraph = connectLayers(lgraph, 'v_flat', 'late_concat/in1');
lgraph = connectLayers(lgraph, 'a_flat', 'late_concat/in2');
net = dlnetwork(lgraph);

%% 5. EĞİTİM DÖNGÜSÜ
fprintf('\n🚀 Eğitim Başlıyor...\n');
trailingAvg = []; trailingAvgSq = [];
trainAccHist = []; testAccHist = []; iteration = 0;

for epoch = 1:maxEpochs
    reset(mbqTrain); shuffle(mbqTrain); epochAcc=0; bCount=0;
    while hasdata(mbqTrain)
        iteration = iteration + 1;
        [XV, XA, YT] = next(mbqTrain);
        
        [grads, loss, YP] = dlfeval(@(n,v,a,y) modelGradients(n,v,a,y), net, XV, XA, YT);
        [net, trailingAvg, trailingAvgSq] = adamupdate(net, grads, trailingAvg, trailingAvgSq, iteration, lr);
        
        [~, pIdx] = max(extractdata(YP),[],1); [~, tIdx] = max(extractdata(YT),[],1);
        epochAcc = epochAcc + sum(pIdx==tIdx)/numel(tIdx); bCount = bCount + 1;
    end
    
    [~, ~, testAcc, ~, ~] = evaluateModel(net, mbqTest, false);
    trainAccHist(end+1) = (epochAcc/bCount)*100; testAccHist(end+1) = testAcc;
    fprintf('Epoch %2d/%d | Train Acc: %5.2f%% | Total Test Acc: %5.2f%%\n', epoch, maxEpochs, trainAccHist(end), testAcc);
end

%% 6. ÇIKTILAR VE GRAFİKLER
fprintf('\n📊 Sonuçlar hesaplanıyor...\n');

fig1 = figure('Visible', 'off'); plot(trainAccHist, 'b-o', 'LineWidth', 2); hold on; plot(testAccHist, 'r-s', 'LineWidth', 2);
title('Training Progress (2D-CNN + 2D-CNN Late Fusion)'); xlabel('Epochs'); ylabel('Accuracy (%)'); legend('Train', 'Test'); grid on;
saveas(fig1, fullfile(saveDir, '1_Progress.png'));

[YPTrain, YTTrain, ~, ~, ~] = evaluateModel(net, mbqTrain, false);
[YPTest, YTTest, totalAcc, visOnlyAcc, audOnlyAcc] = evaluateModel(net, mbqTest, true);

fig2 = figure('Visible', 'off'); confusionchart(categorical(classNames(YTTrain)', classNames), categorical(classNames(YPTrain)', classNames));
title('Train Confusion Matrix'); saveas(fig2, fullfile(saveDir, '2_Conf_Train.png'));

fig3 = figure('Visible', 'off'); confusionchart(categorical(classNames(YTTest)', classNames), categorical(classNames(YPTest)', classNames));
title(sprintf('Test Confusion Matrix (Total Acc: %.2f%%)', totalAcc)); saveas(fig3, fullfile(saveDir, '3_Conf_Test.png'));

fprintf('\n✅ İŞLEM TAMAMLANDI!\n');
fprintf('   -> Total Late Fusion Test Accuracy: %.2f%%\n', totalAcc);
fprintf('   -> Visual-Only (2D-CNN) Accuracy:   %.2f%%\n', visOnlyAcc);
fprintf('   -> Audio-Only (2D-CNN) Accuracy:    %.2f%%\n', audOnlyAcc);
fprintf('   -> Bütün grafikler "%s" dizinine kaydedildi.\n', saveDir);

%% FONKSİYONLAR
function [grads, loss, YP] = modelGradients(net, XV, XA, YT)
    YP = forward(net, XV, XA); 
    loss = crossentropy(YP, YT); 
    grads = dlgradient(loss, net.Learnables);
end

function [V, A, Y] = prepBatch(vC, aC, lC, cNames)
    V = cat(4, vC{:}); A = cat(4, aC{:}); lbl = cat(1, lC{:});
    Y = zeros(numel(cNames), numel(lbl), 'single'); for i = 1:numel(lbl), Y(lbl(i) == cNames, i) = 1; end
end

function [YPred, YTrue, accTotal, accVis, accAud] = evaluateModel(net, mbq, calcSingle)
    reset(mbq); YPred=[]; YTrue=[]; YP_Vis=[]; YP_Aud=[];
    while hasdata(mbq)
        [XV, XA, YT] = next(mbq);
        
        % 1. Total Fusion Tahmini
        out = predict(net, XV, XA); 
        [~, pIdx] = max(extractdata(out),[],1);
        [~, tIdx] = max(extractdata(YT),[],1);
        YPred=[YPred, pIdx]; YTrue=[YTrue, tIdx];
        
        % 2. Tekil Tahminler (Zero-Imputation yöntemi)
        if calcSingle
            % Visual-Only Test (Sesi sıfırla)
            outV = predict(net, XV, zeros(size(XA),'single')); 
            [~, vIdx] = max(extractdata(outV),[],1);
            YP_Vis=[YP_Vis, vIdx];
            
            % Audio-Only Test (Görüntüyü sıfırla)
            outA = predict(net, zeros(size(XV),'single'), XA); 
            [~, aIdx] = max(extractdata(outA),[],1);
            YP_Aud=[YP_Aud, aIdx];
        end
    end
    accTotal = sum(YPred==YTrue)/numel(YTrue)*100;
    if calcSingle
        accVis = sum(YP_Vis==YTrue)/numel(YTrue)*100;
        accAud = sum(YP_Aud==YTrue)/numel(YTrue)*100;
    else
        accVis = 0; accAud = 0;
    end
end

function d = readRGBImage(f, sz)
    path = f; while iscell(path), path = path{1}; end; if isstring(path), path = char(path); end
    img = imread(path); if size(img,3)==1, img=cat(3,img,img,img); end
    d = {single(imresize(img, sz(1:2)))};
end

function d = readMelSpecImage(f, len, fs, sz)
    path = f; while iscell(path), path = path{1}; end; if isstring(path), path = char(path); end
    [a, afs] = audioread(path); 
    if afs ~= fs, a = resample(a, fs, afs); end
    if size(a, 2) > 1, a = mean(a, 2); end
    if numel(a) > len, a = a(1:len); else, a = [a; zeros(len-numel(a), 1)]; end
    
    S = log10(melSpectrogram(a, fs, 'NumBands', sz(1)) + 1e-6); 
    S = (S - mean(S(:))) / (std(S(:)) + 1e-8);
    d = {reshape(single(imresize(S, sz(1:2))), [sz(1) sz(2) 1])};
end