% Human action recognition
% Test dataset : MSR Action3D
% Cross Subject Test
% by Chen Chen, The University of Texas at Dallas
% chenchen870713@gmail.com

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% Subjects for training and testing are FIXED.
%%%%%%% Subject 1 3 5 7 9 as training subjects, the rest as testing subjects
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% profile on
file_dir = 'MSR-Action3D\';
ActionNum = ['a01','a02', 'a03', 'a04', 'a05', 'a06', 'a07', 'a08', 'a09','a10','a11','a12','a13','a14','a15','a16','a17','a18','a19','a20';]
            
NumAct = 20;          % number of actions in each subset
row = 240;
col = 320;
max_subject = 10;    % maximum number of subjects for one action
max_experiment = 3;  % maximum number of experiments performed by one subject
lambda = 0.001;
frame_remove = 5;     % remove the first and last five frames (mostly the subject is in stand-still position in these frames)

ActionSet = 'AS1';
fprintf('Action set: %s\n', ActionSet);

switch ActionSet
    case 'AS1'
        subset = 1;
%         fix_size_front = round([100;50]/2); fix_size_side = round([100;82]/2); fix_size_top = round([82;47]/2);
        fix_size_front = [100;50]; fix_size_side = [100;82]; fix_size_top = [82;47];
        % the fixed size of each projection view is calculated as the
        % average size of DMMs of all samples, here we did not optimize the
        % sizes.
        % 
    case 'AS2'
        subset = 2;
        %fix_size_front = round([102;51]/2); fix_size_side = round([103;67]/2); fix_size_top = round([67;51]/2);
        fix_size_front = [102;51]; fix_size_side = [103;67]; fix_size_top = [67;51];
    case 'AS3'
        subset = 3;
        %fix_size_front = round([104;53]/2); fix_size_side = round([104;84]/2); fix_size_top = round([84;53]/2);
        fix_size_front = [104;53]; fix_size_side = [104;84]; fix_size_top = [84;53];
end
D = prod(fix_size_front)+prod(fix_size_side)+prod(fix_size_top);
fix_size = [fix_size_front;fix_size_side;fix_size_top];

TargetSet = ActionNum(subset,:);
TotalNum = max_subject*max_experiment*NumAct; % assume 10 subjects, 3 experiments per subject for each action
TotalFeature = zeros(D,TotalNum);

%% Generate DMM for all depth sequences in one action set

subject_ind = cell(1,NumAct);
OneActionSample = zeros(1,NumAct);
cnt=1;
% profile on
for i = 1:NumAct  
    action = TargetSet((i-1)*3+1:i*3);
    action_dir = strcat(file_dir,action,'\');
    fpath = fullfile(action_dir, '*.mat');
    depth_dir = dir(fpath);
    ind = zeros(1,length(depth_dir));
    for j = 1:length(depth_dir)
        depth_name = depth_dir(j).name;
        sub_num = str2double(depth_name(6:7));
        ind(j) = sub_num;
        load(strcat(action_dir,depth_name));
        depth = depth(:,:,frame_remove+1:end-frame_remove);
        [front, side, top] = depth_projection1(depth);   
        front = resize_feature1(front,fix_size_front);
        side  = resize_feature1(side,fix_size_side);
        top   = resize_feature1(top,fix_size_top);   
        TotalFeature(:,sum(OneActionSample)+j) = [front;side;top];
        strval=[action(2:3) depth_name(6:7) depth_name(6:7)];
        tottrainlabel(cnt,1)=str2double(action(2:3));
        cnt=cnt+1;
    end
    OneActionSample(i) = length(depth_dir);
    subject_ind{i} = ind;
    
end
% profileStruct = profile('info');
% [flopTotal,Details]  = FLOPS('depth_projection1','depthprojection1',profileStruct);%
TotalFeature = TotalFeature(:,1:sum(OneActionSample));



%% Generate training and testing data

% train_index = [2 4 6 8 10];
train_index = [1 3 5 7 9];
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
Dim = size(F_train,2) - 20;  % AS1:12 AS2:20 AS3:7 for 1 3 5 7 9 as training

% Original Eigenface PCA
profile on
disc_set = Eigenface_f(single(F_train),Dim);
profileStruct = profile('info');
[flopTotal,Details]  = FLOPS('Eigenface_f','eigenfacef',profileStruct);%
F_train = disc_set'*F_train;
F_test  = disc_set'*F_test;
F_train = F_train./(repmat(sqrt(sum(F_train.*F_train)), [Dim,1]));
F_test  = F_test./(repmat(sqrt(sum(F_test.*F_test)), [Dim,1]));

%% Reduced Basics Decomposition

% num=103;  %%AS3=103 AS2=122  AS1=120
% m = size(F_train,1);
% n = size(F_train,2);
% compratio = 1;  %%AS3=1.1 %%AS2=1.5
% num = floor(min(m,n)/compratio);
% basnum = 100;
% [nr,nc] = size(F_train(:,:,1));
% profile on
% [a, b] = RBD(single(F_train),1e-20);   %%AS2 AS1
% [a, b] = RBD(single(F_train),1e-20,num);  %%AS3  num=103;
% %[a, b] = RBD(single(F_train),1e-20,basnum,1,zeros(nr*nc,1));
% profileStruct = profile('info');
% [flopTotal,Details]  = FLOPS('RBD','RBDmat',profileStruct);%
% F_train = a'*F_train;
% F_test  = a'*F_test;

%% Testing

%////////////////////////////////////////////////////////////////////%    
%         Tikhonov regularized Collaborative Classifier              %
%////////////////////////////////////////////////////////////////////%

% % profile on
% label = L2_CRC(F_train, F_test, F_train_size, NumAct, lambda);
% % profileStruct = profile('info');
% % [flopTotal,Details]  = FLOPS('L2_CRC','exampleScriptMAT',profileStruct);%
% [confusion, accuracy, CR, FR] = confusion_matrix(label, F_test_size);
% fprintf('Accuracy = %f\n', accuracy);
% profsave
% profile off

%% Collaborative Representation Root Least Squares
% kappa             =   [0.001];    
% par.nClass        =   100;                 % the number of classes in the subset of AR database
% par.nDim          =   Dim;                 % the eigenfaces dimension
% 
% tr_dat = F_train;
% tt_dat = F_test;
% trls     =   otrainlabel(otrainlabel<=par.nClass);
% ttls     =   otestlabel(otestlabel<=par.nClass);
% 
% %projection matrix computing
% Proj_M = inv(tr_dat'*tr_dat+kappa*eye(size(tr_dat,2)))*tr_dat';
% 
% %-------------------------------------------------------------------------
% %testing
% ID = [];
% profile on
% for indTest = 1:size(tt_dat,2)
%     [id]    = CRC_RLS(tr_dat,Proj_M,tt_dat(:,indTest),trls);
%     ID      =   [ID id];
% end
% profileStruct = profile('info');
% [flopTotal,Details]  = FLOPS('CRC_RLS','crcrls',profileStruct);%
% cornum      =   sum(ID==ttls);
% Rec         =   [cornum/length(ttls)]; % recognition rate
%  fprintf(['recogniton rate is ' num2str(Rec)]);
% % [confusion, accur_CRT, TPR, FPR] = confusion_matrix_wei(ID', F_test_size);
% cMat1 = confusionmat(otestlabel,ID');
% % profsave
% % profile off




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
params.class_num  =  30; % Ideal 50, 30
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

accuracy = (sum(pred_tt_label == data.tt_label)) / length(data.tt_label);
cMat1 = confusionmat(otestlabel,pred_tt_label');

fprintf(['\nThe accuracy on the ', params.dataset_name, ' dataset ',' with gamma=',num2str(params.gamma), ' and lambda=',num2str(params.lambda), ' is ', num2str(roundn(accuracy,-3)),'\n'])
% profsave
% profile off


