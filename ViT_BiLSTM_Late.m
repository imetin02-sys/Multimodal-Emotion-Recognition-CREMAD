%% 1. MODEL: 2D-CNN (Visual) + 2D-CNN (Mel-Spec Audio) EARLY FUSION
clear; clc; close all;

%% 1. YOLLAR VE HİPERPARAMETRELER
imgBaseDir = 'C:\Users\STB\Desktop\DATASET\CREMAD\Image'; 
audDir     = 'C:\Users\STB\Desktop\DATASET\CREMAD\Speech';      
saveDir    = 'C:\Users\STB\Desktop\Araştırma\Model1_2D_2D_Early';
if ~exist(saveDir, 'dir'), mkdir(saveDir); end

visImgSize = [224 224 3];  
melImgSize = [128 128 1];  
audLen = 48000; fs = 16000;
batchSize = 16; lr = 0.0001; maxEpochs = 40;

fprintf('\n🚀 MODEL 1: 2D-CNN + 2D-CNN EARLY FUSION BAŞLIYOR...\n\n');

%% 2. ACTOR BASED SPLIT (%80-%20)
classFolders = dir(imgBaseDir);
classFolders = classFolders([classFolders.isdir] & ~ismember({classFolders.name}, {'.', '..'}));
classNames = {classFolders.name};
numClasses = numel(classNames);

trainImg = {}; trainAud = {}; trainLbl = [];
testImg  = {}; testAud  = {}; testLbl  = [];

% --- HATA ÇÖZÜMÜ: Ses dosyalarını alt klasörlerde dahi olsalar bul ---
fprintf('🔍 Ses dosyaları taranıyor...\n');
allAud = dir(fullfile(audDir, '**', '*.wav'));
audNames = {allAud.name};
audFolders = {allAud.folder};

for c = 1:numClasses
    cName = classNames{c};
    imgsInClass = dir(fullfile(imgBaseDir, cName, '*.jpg'));
    
    % Bu sınıftaki eşsiz videoları (baseName) bul
    baseNames = cell(numel(imgsInClass), 1);
    for i = 1:numel(imgsInClass)
        parts = split(imgsInClass(i).name, '_frame');
        baseNames{i} = parts{1};
    end
    uniqueVideos = unique(baseNames);
    uniqueVideos = uniqueVideos(randperm(numel(uniqueVideos))); % Karıştır
    
    % Videoların %80'ini Train, %20'sini Test yap
    nTrain = round(0.80 * numel(uniqueVideos));
    trainVids = uniqueVideos(1:nTrain);
    
    for i = 1:numel(imgsInClass)
        imgFile = fullfile(imgsInClass(i).folder, imgsInClass(i).name);
        
        % Ses dosyasının tam yolunu bul
        targetAudName = [baseNames{i}, '.wav'];
        matchIdx = find(strcmpi(audNames, targetAudName), 1);
        
        if ~isempty(matchIdx)
            audFile = fullfile(audFolders{matchIdx}, audNames{matchIdx});
            
            if ismember(baseNames{i}, trainVids)
                trainImg{end+1} = imgFile; trainAud{end+1} = audFile; trainLbl = [trainLbl; c];
            else
                testImg{end+1}  = imgFile; testAud{end+1}  = audFile; testLbl  = [testLbl; c];
            end
        end
    end
    fprintf('📁 Sınıf: %s -> %d Train, %d Test Karesi Ayrıldı.\n', cName, sum(trainLbl==c), sum(testLbl==c));
end

% --- GÜVENLİK KONTROLÜ (Expected at most 0 inputs hatasını önler) ---
if isempty(trainImg)
    error('\n❌ HATA: Hiçbir ses ve görüntü çifti eşleştirilemedi! Lütfen "audDir" (Ses) yolunu kontrol edin.\n');
end

YTrain = categorical(classNames(trainLbl)', classNames);
YTest  = categorical(classNames(testLbl)', classNames);

%% 3. DATASTORE KURULUMU
dsVisTrain = transform(arrayDatastore(trainImg', 'IterationDimension', 1), @(f) readRGBImage(f, visImgSize));
dsVisTest  = transform(arrayDatastore(testImg', 'IterationDimension', 1),  @(f) readRGBImage(f, visImgSize));
dsAudTrain = transform(arrayDatastore(trainAud', 'IterationDimension', 1), @(f) readMelSpecImage(f, audLen, fs, melImgSize));
dsAudTest  = transform(arrayDatastore(testAud', 'IterationDimension', 1),  @(f) readMelSpecImage(f, audLen, fs, melImgSize));

dsTrain = combine(dsVisTrain, dsAudTrain, arrayDatastore(YTrain));
dsTest  = combine(dsVisTest, dsAudTest, arrayDatastore(YTest));

mbqTrain = minibatchqueue(dsTrain, 3, 'MiniBatchSize', batchSize, 'MiniBatchFcn', @(v,a,l) prepBatch(v,a,l,classNames), 'MiniBatchFormat', {'SSCB', 'SSCB', 'CB'});
mbqTest  = minibatchqueue(dsTest,  3, 'MiniBatchSize', batchSize, 'MiniBatchFcn', @(v,a,l) prepBatch(v,a,l,classNames), 'MiniBatchFormat', {'SSCB', 'SSCB', 'CB'});

%% 4. MİMARİ: EARLY FUSION
lgraph = layerGraph();

visBranch = [
    imageInputLayer(visImgSize, 'Name', 'in_vis', 'Normalization', 'zscore')
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'v_c1')
    reluLayer('Name', 'v_r1')
    maxPooling2dLayer(4, 'Stride', 4, 'Name', 'v_p1')
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'v_c2')
    reluLayer('Name', 'v_r2')
    globalAveragePooling2dLayer('Name', 'v_gap')
    flattenLayer('Name', 'v_flat')
]; 
lgraph = addLayers(lgraph, visBranch);

audBranch = [
    imageInputLayer(melImgSize, 'Name', 'in_aud', 'Normalization', 'zscore')
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'a_c1')
    reluLayer('Name', 'a_r1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'a_p1')
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'a_c2')
    reluLayer('Name', 'a_r2')
    globalAveragePooling2dLayer('Name', 'a_gap')
    flattenLayer('Name', 'a_flat')
]; 
lgraph = addLayers(lgraph, audBranch);

shared = [
    concatenationLayer(1, 2, 'Name', 'concat')
    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'r1')
    dropoutLayer(0.5, 'Name', 'drop')
    fullyConnectedLayer(numClasses, 'Name', 'fc_out')
    softmaxLayer('Name', 'sm')
]; 
lgraph = addLayers(lgraph, shared);

lgraph = connectLayers(lgraph, 'v_flat', 'concat/in1');
lgraph = connectLayers(lgraph, 'a_flat', 'concat/in2');
net = dlnetwork(lgraph);

%% 5. EĞİTİM
fprintf('\n🚀 Eğitim Başlıyor...\n');
trailingAvg = []; trailingAvgSq = [];
trainAccHist = []; testAccHist = []; iteration = 0;

for epoch = 1:maxEpochs
    reset(mbqTrain); shuffle(mbqTrain); epochAcc=0; bCount=0;
    while hasdata(mbqTrain)
        iteration = iteration + 1;
        [XV, XA, YT] = next(mbqTrain);
        [grads, state, loss, YP] = dlfeval(@(n,v,a,y) modelGradients(n,v,a,y), net, XV, XA, YT);
        net.State = state;
        [net, trailingAvg, trailingAvgSq] = adamupdate(net, grads, trailingAvg, trailingAvgSq, iteration, lr);
        
        [~, pIdx] = max(extractdata(YP),[],1); [~, tIdx] = max(extractdata(YT),[],1);
        epochAcc = epochAcc + sum(pIdx==tIdx)/numel(tIdx); bCount = bCount + 1;
    end
    
    [~, ~, testAcc, ~] = evaluateModel(net, mbqTest, false);
    trainAccHist(end+1) = (epochAcc/bCount)*100; testAccHist(end+1) = testAcc;
    fprintf('Epoch %2d/%d | Train Acc: %5.2f%% | Total Test Acc: %5.2f%%\n', epoch, maxEpochs, trainAccHist(end), testAcc);
end

%% 6. DEĞERLENDİRME VE GRAFİKLER
fig1 = figure('Visible', 'off'); plot(trainAccHist, 'b-o', 'LineWidth', 2); hold on; plot(testAccHist, 'r-s', 'LineWidth', 2);
title('Training Progress'); xlabel('Epochs'); ylabel('Accuracy (%)'); legend('Train', 'Test'); grid on; 
saveas(fig1, fullfile(saveDir, '1_Progress.png'));

[YPTrain, YTTrain, ~, ~] = evaluateModel(net, mbqTrain, false);
[YPTest, YTTest, totalAcc, visOnlyAcc] = evaluateModel(net, mbqTest, true);

fig2 = figure('Visible', 'off'); confusionchart(categorical(classNames(YTTrain)', classNames), categorical(classNames(YPTrain)', classNames));
title('Train Confusion Matrix'); saveas(fig2, fullfile(saveDir, '2_Conf_Train.png'));

fig3 = figure('Visible', 'off'); confusionchart(categorical(classNames(YTTest)', classNames), categorical(classNames(YPTest)', classNames));
title(sprintf('Test Confusion Matrix (Acc: %.2f%%)', totalAcc)); saveas(fig3, fullfile(saveDir, '3_Conf_Test.png'));

fprintf('\n✅ SONUÇLAR:\n');
fprintf('   -> Total Early Fusion Test Accuracy: %.2f%%\n', totalAcc);
fprintf('   -> Visual-Only (Sıfırlanmış Ses) Accuracy: %.2f%%\n', visOnlyAcc);
fprintf('   -> Bütün grafikler "%s" dizinine kaydedildi.\n', saveDir);

%% FONKSİYONLAR
function [grads, state, loss, YP] = modelGradients(net, XV, XA, YT)
    [YP, state] = forward(net, XV, XA); 
    loss = crossentropy(YP, YT); 
    grads = dlgradient(loss, net.Learnables);
end

function [V, A, Y] = prepBatch(vC, aC, lC, cNames)
    V = cat(4, vC{:}); 
    A = cat(4, aC{:}); 
    lbl = cat(1, lC{:});
    Y = zeros(numel(cNames), numel(lbl), 'single'); 
    for i = 1:numel(lbl)
        Y(lbl(i) == cNames, i) = 1; 
    end
end

function [YPred, YTrue, accTotal, accVisOnly] = evaluateModel(net, mbq, calcVisOnly)
    reset(mbq); YPred = []; YTrue = []; YPredVis = [];
    while hasdata(mbq)
        [XV, XA, YT] = next(mbq);
        out = predict(net, XV, XA); 
        [~, pIdx] = max(extractdata(out), [], 1);
        [~, tIdx] = max(extractdata(YT), [], 1);
        YPred = [YPred, pIdx]; 
        YTrue = [YTrue, tIdx];
        
        if calcVisOnly % Sıfırlama (Zero-Imputation) Tekniği ile Visual-Only başarısı
            outVis = predict(net, XV, zeros(size(XA), 'single'));
            [~, pVIdx] = max(extractdata(outVis), [], 1); 
            YPredVis = [YPredVis, pVIdx];
        end
    end
    accTotal = sum(YPred == YTrue) / numel(YTrue) * 100;
    if calcVisOnly
        accVisOnly = sum(YPredVis == YTrue) / numel(YTrue) * 100; 
    else
        accVisOnly = 0; 
    end
end

function d = readRGBImage(f, sz)
    % Veri tipi ne olursa olsun (cell, string, char) dosya yolunu güvenle çıkar
    filePath = f;
    while iscell(filePath), filePath = filePath{1}; end
    if isstring(filePath), filePath = char(filePath); end
    
    img = imread(filePath); 
    if size(img, 3) == 1, img = cat(3, img, img, img); end
    d = {single(imresize(img, sz(1:2)))}; 
end

function d = readMelSpecImage(f, len, fs, sz)
    % Veri tipi ne olursa olsun (cell, string, char) dosya yolunu güvenle çıkar
    filePath = f;
    while iscell(filePath), filePath = filePath{1}; end
    if isstring(filePath), filePath = char(filePath); end
    
    [a, afs] = audioread(filePath); 
    if afs ~= fs, a = resample(a, fs, afs); end
    if size(a, 2) > 1, a = mean(a, 2); end
    if numel(a) > len, a = a(1:len); else, a = [a; zeros(len-numel(a), 1)]; end
    
    S = log10(melSpectrogram(a, fs, 'NumBands', sz(1)) + 1e-6); 
    S = (S - mean(S(:))) / (std(S(:)) + 1e-8);
    d = {reshape(single(imresize(S, sz(1:2))), [sz(1) sz(2) 1])};
end
