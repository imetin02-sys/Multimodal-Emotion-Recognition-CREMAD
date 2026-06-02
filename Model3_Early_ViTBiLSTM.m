%% MODEL 3: ViT-Mixer (Visual) + BiLSTM (Audio MFCC) EARLY FUSION (RASTGELE BÖLME)
clear; clc; close all;

%% 1. YOLLAR VE PARAMETRELER
imgBaseDir = 'C:\Users\STB\Desktop\DATASET\CREMAD\Image'; 
audDir     = 'C:\Users\STB\Desktop\DATASET\CREMAD\Speech';      
saveDir    = 'C:\Users\STB\Desktop\Araştırma\Model3_ViT_BiLSTM_Early';
if ~exist(saveDir, 'dir'), mkdir(saveDir); end

visImgSize = [224 224 3];  
audSeqLen = 100;     % Zaman adımı (Sequence Length)
numMfcc = 13;        % MFCC Kanal Sayısı

audLen = 48000; fs = 16000;
batchSize = 16; lr = 0.0001; maxEpochs = 40;

fprintf('\n🚀 MODEL 3: ViT + BiLSTM EARLY FUSION BAŞLIYOR (RASTGELE BÖLME)...\n\n');

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

%% 3. DATASTORE VE MİNİBATCHQUEUE (ZAMAN SERİSİ FORMATI İLE)
dsVisTrain = transform(arrayDatastore(trainImg', 'IterationDimension', 1), @(f) readRGBImage(f, visImgSize));
dsVisTest  = transform(arrayDatastore(testImg', 'IterationDimension', 1),  @(f) readRGBImage(f, visImgSize));

dsAudTrain = transform(arrayDatastore(trainAud', 'IterationDimension', 1), @(f) readMfccSequence(f, audLen, fs, audSeqLen, numMfcc));
dsAudTest  = transform(arrayDatastore(testAud', 'IterationDimension', 1),  @(f) readMfccSequence(f, audLen, fs, audSeqLen, numMfcc));

dsTrain = combine(dsVisTrain, dsAudTrain, arrayDatastore(YTrain));
dsTest  = combine(dsVisTest, dsAudTest, arrayDatastore(YTest));

% DİKKAT: BiLSTM için ses formatı CBT'dir!
mbqTrain = minibatchqueue(dsTrain, 3, 'MiniBatchSize', batchSize, 'MiniBatchFcn', @(v,a,l) prepBatchSeq(v,a,l,classNames), 'MiniBatchFormat', {'SSCB', 'CBT', 'CB'});
mbqTest  = minibatchqueue(dsTest,  3, 'MiniBatchSize', batchSize, 'MiniBatchFcn', @(v,a,l) prepBatchSeq(v,a,l,classNames), 'MiniBatchFormat', {'SSCB', 'CBT', 'CB'});

%% 4. MİMARİ: ViT + BiLSTM (EARLY FUSION)
lgraph = layerGraph();

% --- Görsel Dal (ViT Patch-Mixer Simülasyonu) ---
visBranch = [
    imageInputLayer(visImgSize, 'Name', 'in_vis', 'Normalization', 'zscore')
    convolution2dLayer(16, 64, 'Stride', 16, 'Padding', 'same', 'Name', 'v_patch') 
    reluLayer('Name', 'v_r1')
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'v_mix')
    reluLayer('Name', 'v_r2')
    globalAveragePooling2dLayer('Name', 'v_gap')
    flattenLayer('Name', 'v_flat')
]; 
lgraph = addLayers(lgraph, visBranch);

% --- İşitsel Dal (BiLSTM Zaman Serisi) ---
audBranch = [
    sequenceInputLayer(numMfcc, 'Name', 'in_aud', 'Normalization', 'zscore')
    % Geçmiş ve geleceği okuyan çift yönlü bellek
    bilstmLayer(128, 'OutputMode', 'last', 'Name', 'a_bilstm')
    % DİKKAT: BiLSTM 'last' modunda zaten düz(1D) çıkar, flatten konulmaz!
]; 
lgraph = addLayers(lgraph, audBranch);

% --- Erken Füzyon Birleştirme Masası (Ortak Karar Ağı) ---
shared = [
    concatenationLayer(1, 2, 'Name', 'early_concat')
    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'r1')
    dropoutLayer(0.5, 'Name', 'drop')
    fullyConnectedLayer(numClasses, 'Name', 'fc_out')
    softmaxLayer('Name', 'sm')
]; 
lgraph = addLayers(lgraph, shared);

% Özellikleri Early Fusion masasına bağla
lgraph = connectLayers(lgraph, 'v_flat', 'early_concat/in1');
lgraph = connectLayers(lgraph, 'a_bilstm', 'early_concat/in2'); 
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
        
        % DİKKAT: State (hafıza) sızıntısı engellendi.
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
title('Training Progress (ViT + BiLSTM Early Fusion)'); xlabel('Epochs'); ylabel('Accuracy (%)'); legend('Train', 'Test'); grid on;
saveas(fig1, fullfile(saveDir, '1_Progress.png'));

[YPTrain, YTTrain, ~, ~, ~] = evaluateModel(net, mbqTrain, false);
[YPTest, YTTest, totalAcc, visOnlyAcc, audOnlyAcc] = evaluateModel(net, mbqTest, true);

fig2 = figure('Visible', 'off'); confusionchart(categorical(classNames(YTTrain)', classNames), categorical(classNames(YPTrain)', classNames));
title('Train Confusion Matrix'); saveas(fig2, fullfile(saveDir, '2_Conf_Train.png'));

fig3 = figure('Visible', 'off'); confusionchart(categorical(classNames(YTTest)', classNames), categorical(classNames(YPTest)', classNames));
title(sprintf('Test Confusion Matrix (Total Acc: %.2f%%)', totalAcc)); saveas(fig3, fullfile(saveDir, '3_Conf_Test.png'));

fprintf('\n✅ İŞLEM TAMAMLANDI!\n');
fprintf('   -> Total Early Fusion Test Accuracy: %.2f%%\n', totalAcc);
fprintf('   -> Visual-Only (ViT) Accuracy: %.2f%%\n', visOnlyAcc);
fprintf('   -> Audio-Only (BiLSTM) Accuracy: %.2f%%\n', audOnlyAcc);
fprintf('   -> Sonuçlar "%s" dizinine kaydedildi.\n', saveDir);

%% FONKSİYONLAR
function [grads, loss, YP] = modelGradients(net, XV, XA, YT)
    YP = forward(net, XV, XA); 
    loss = crossentropy(YP, YT); 
    grads = dlgradient(loss, net.Learnables);
end

function [V, A, Y] = prepBatchSeq(vC, aC, lC, cNames)
    V = cat(4, vC{:}); 
    % Sesi C x T x B formatında alıp, CBT (C x B x T) formatına döndürüyoruz
    A_temp = cat(3, aC{:}); 
    A = permute(A_temp, [1 3 2]); 
    lbl = cat(1, lC{:});
    Y = zeros(numel(cNames), numel(lbl), 'single'); for i = 1:numel(lbl), Y(lbl(i) == cNames, i) = 1; end
end

function [YPred, YTrue, accTotal, accVis, accAud] = evaluateModel(net, mbq, calcSingle)
    reset(mbq); YPred=[]; YTrue=[]; YP_Vis=[]; YP_Aud=[];
    while hasdata(mbq)
        [XV, XA, YT] = next(mbq);
        
        % 1. Tam Füzyon Testi
        out = predict(net, XV, XA); [~, pIdx] = max(extractdata(out),[],1);
        [~, tIdx] = max(extractdata(YT),[],1);
        YPred=[YPred, pIdx]; YTrue=[YTrue, tIdx];
        
        % 2. Sıfırlama (Zero-Imputation) Testleri
        if calcSingle
            outV = predict(net, XV, zeros(size(XA),'single')); [~, vIdx] = max(extractdata(outV),[],1);
            YP_Vis=[YP_Vis, vIdx];
            outA = predict(net, zeros(size(XV),'single'), XA); [~, aIdx] = max(extractdata(outA),[],1);
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
    path = f; while iscell(path), path = path{1}; end
    img = imread(char(path)); if size(img,3)==1, img=cat(3,img,img,img); end
    d = {single(imresize(img, sz(1:2)))};
end

function d = readMfccSequence(f, len, fs, tLen, numCh)
    path = f; while iscell(path), path = path{1}; end
    [a, afs] = audioread(char(path));
    if afs~=fs, a=resample(a,fs,afs); end; if size(a,2)>1, a=mean(a,2); end
    if numel(a)>len, a=a(1:len); else, a=[a; zeros(len-numel(a),1)]; end
    
    c = single(mfcc(a, fs, 'NumCoeffs', numCh-1)); 
    c = (c - mean(c(:))) / (std(c(:)) + 1e-8);
    
    if size(c,1)<tLen
        c = [c; zeros(tLen-size(c,1), size(c,2), 'single')]; 
    else
        c = c(1:tLen,:); 
    end
    d = {c'};
end