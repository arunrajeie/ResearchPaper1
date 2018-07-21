%% The Following is the most important Test which consider all combinations of Training and Tesing Sets
%% Leave One Subject Out Test was also included


load ('Totalfeaturefull');
train_total = [1 2 3 4 5 6 7 8 9 10];
k1=5;  %% Determines the number of training and testing subjects
train_index1 = combnk(train_total,k1);               %% for 252 Combo
% train_index = [2 4 6 8 10];
% train_index = [1 3 5 7 9];
% train_index = [1 2 3 4 5 6 7 8 9 10];
[r1 m1] = size(train_index1);
accuracy = zeros(1,r1);

for row1 = 1:1:r1
    train_index = train_index1(row1,:);


F_train_size = zeros(1,NumAct);
F_test_size  = zeros(1,NumAct);
F_train = [];
F_test = [];
otrainlabel=[];
otestlabel=[];
trainlabel=[];
testlabel=[];

count = 0;
for i = 1:NumAct 
    ID = subject_ind{i};
    F = TotalFeature(:,count+1:count+OneActionSample(i));
    ttlabel=tottrainlabel(count+1:count+OneActionSample(i));
    for k = 1:length(train_index)
        ID(ID==train_index(k)) = 0;
    end
    F_train = [F_train F(:,ID==0)];
    trainlabel=[trainlabel 
            ttlabel(ID==0)];
    F_test  = [F_test F(:,ID>0)];
    testlabel=[testlabel ttlabel(ID>0)'];
    F_train_size(i) = sum(ID==0);
    F_test_size(i)  = size(F,2) - F_train_size(i);
    count = count + OneActionSample(i);
end
otrainlabel=trainlabel;
otestlabel=testlabel;

%%%%% PCA on training samples and test samples
Dim = size(F_train,2);  % AS1:12 AS2:20 AS3:7 for 1 3 5 7 9 as training
                             % AS1:20 AS2:7 AS3:24 for 1 3 5 7 9 as training
% Original Eigenface PCA

% disc_set = Eigenface_f(single(F_train),Dim);
% F_train = disc_set'*F_train;
% F_test  = disc_set'*F_test;
% F_train = F_train./(repmat(sqrt(sum(F_train.*F_train)), [Dim,1]));
% F_test  = F_test./(repmat(sqrt(sum(F_test.*F_test)), [Dim,1]));

%% Reduced Basics Decomposition
% 
num=Dim;  %%AS3=103 AS2=122  AS1=120
density = 0.01;
Norm1 = ones(Dim2);         %% All ones Matrix Norm
% Norm1 = diag(diag(ones(Dim2)));   %% Identity Matrix Norm
% Dim22 = randi(Dim2,Dim2);
% Norm1 = diag((diag(Dim22)));   %% Diagonal Matrix Norm
% Norm1 = generateSPDmatrix(Dim2);  %% SPD Matrix Norm1 
% % Norm1 = full(Norm1);
% m = size(F_train,1);
% n = size(F_train,2);
% compratio = 1.1;  %%AS1=1,AS3=1.2    
% num = floor(min(m,n)/compratio);
% basnum = 120;
% [nr,nc] = size(F_train(:,:,1));
tic 
% [a, b] = RBD(single(F_train));   %%AS2 AS1
% [a, b] = RBD(single(F_train),1e-20);   %%AS2 AS1

[a, b] = RBD(single(F_train),1e-20,num);  %%AS3  num=103;
% 
F_train = a'*F_train;
F_test  = a'*F_test;
% F_train(find(isnan(F_train)))=0;
% F_test(find(isnan(F_test)))=0;


%% Testing

%////////////////////////////////////////////////////////////////////%    
%         Tikhonov regularized Collaborative Classifier              %
%////////////////////////////////////////////////////////////////////%

% % profile on
% label = L2_CRC(F_train, F_test, F_train_size, NumAct, lambda);
% % profileStruct = profile('info');
% % [flopTotal,Details]  = FLOPS('L2_CRC','exampleScriptMAT',profileStruct);%
% [confusion, accuracy(row1), CR, FR] = confusion_matrix(label, F_test_size);
% fprintf('Accuracy = %f\n', accuracy(row1));
% clear train_index;
% profsave
% profile off

%%  Probability Based Collaborative Representation Classifier

data.tr_descr  = F_train;
data.tt_descr  = F_test;
data.tr_label   = otrainlabel';
data.tt_label  = otestlabel;
dataset.label = tottrainlabel;
% class_num = length(unique(data.tr_label));
 
params.gamma     =  [1e-2];
% params.lambda     =  [1e-1];
params.lambda     =  [1e-1];   %% good accuracy
params.class_num  =  20; %% 14 for gamma=0, %% AS1 = 20, AS2 = 30, AS3 = 20 Represent the number of training samples rmin in Big-O
% params.class_num  =  30; 
params.dataset_name = 'MSR-Action3D';

params.model_type =  'ProCRC';

% profile on
Alpha = ProCRC(data, params);
% profileStruct = profile('info');
% [flopTotal,Details]  = FLOPS('ProCRC','ProCRC',profileStruct);%

% profile on
[pred_tt_label, ~] = ProMax(Alpha, data, params);
% profileStruct = profile('info');
% [flopTotal,Details]  = FLOPS('ProMax','exampleScriptMAT',profileStruct);%

accuracy(row1) = (sum(pred_tt_label == data.tt_label)) / length(data.tt_label);
cMat1 = confusionmat(otestlabel,pred_tt_label');

fprintf(['\nThe accuracy on the ', params.dataset_name, ' dataset ',' with gamma=',num2str(params.gamma), ' and lambda=',num2str(params.lambda), ' is ', num2str(roundn(accuracy(row1),-3)),'\n'])
clear train_index;

% if verbose
%         fprintf('Trial %d accuracy = %f\n', row1, accuracy(row1));
% end
end



fprintf('Average accuracy = %f; std = %f \n', mean(accuracy), std(accuracy));







