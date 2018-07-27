% Human action recognition
% Test dataset : MSR Action3D
% Cross Subject Test
% by Chen Chen, The University of Texas at Dallas
% chenchen870713@gmail.com
%% Some Part of the code is acquired from the above author, while the major modification
%% is done by the Author of this publication arunrajeie@gmail.com.

file_dir = '../data/MSR-Action3D/';

% ActionNum = ['a01','a02','a03','a04','a05','a06','a07','a08','a09','a10','a11','a12','a13'...
%     'a14','a15','a16','a17','a18','a19','a20'];
         
 ActionNum = ['a02', 'a03', 'a05', 'a06', 'a10', 'a13', 'a18', 'a20'; % first row corresponds to action subset 'AS1'
              'a01', 'a04', 'a07', 'a08', 'a09', 'a11', 'a14', 'a12'; % second row corresponds to action subset 'AS2'
              'a06', 'a14', 'a15', 'a16', 'a17', 'a18', 'a19', 'a20']; % third row corresponds to action subset 'AS3'
            
NumAct = 8;          % 8, 20 number of actions in each subset
row = 240;
col = 320;
max_subject = 10;    % maximum number of subjects for one action
max_experiment = 3;  % maximum number of experiments performed by one subject
lambda = 0.001;
frame_remove = 5;     % remove the first and last five frames (mostly the subject is in stand-still position in these frames)
% frame_remove = 2;

%% Fixed size is preferrable to avoid any manual tuning

fix_size_front = [102,54]; fix_size_side = [102,75]; fix_size_top = [75,54];  %%%% DMM sizes (fixed size)

ActionSet = 'AS1';
fprintf('Action set: %s\n', ActionSet);

switch ActionSet
    case 'AS1'
        subset = 1;
                 fix_size_front = [100;50]; fix_size_side = [100;82]; fix_size_top = [82;47]; 
%                 fix_size_front = round([100;50]/2); fix_size_side = round([100;82]/2); fix_size_top = round([82;47]/2);
%                fix_size_front = [102,54]; fix_size_side = [102,75]; fix_size_top = [75,54];      

        % the fixed size of each projection view is calculated as the
        % average size of DMMs of all samples, here we did not optimize the
        % sizes.
        % 
    case 'AS2'
        subset = 2;
        %fix_size_front = round([102;51]/2); fix_size_side = round([103;67]/2); fix_size_top = round([67;51]/2);
        fix_size_front = [102;51]; fix_size_side = [103;67]; fix_size_top = [67;51]; fix_size_temp1 = [100;100];
    case 'AS3'
        subset = 3;
        %fix_size_front = round([104;53]/2); fix_size_side = round([104;84]/2); fix_size_top = round([84;53]/2);
        fix_size_front = [104;53]; fix_size_side = [104;84]; fix_size_top = [84;53]; fix_size_temp1 = [100;100];
end

D = prod(fix_size_front)+prod(fix_size_side)+prod(fix_size_top);
% D = prod(fix_size_front)+prod(fix_size_side)+prod(fix_size_top)+prod(fix_size_temp1);
% fix_size = [fix_size_front;fix_size_side;fix_size_top];

TargetSet = ActionNum(subset,:);
TotalNum = max_subject*max_experiment*NumAct; % assume 10 subjects, 3 experiments per subject for each action
TotalFeature = zeros(D,TotalNum);

%% Generate DMM for all depth sequences in one action set

subject_ind = cell(1,NumAct);
OneActionSample = zeros(1,NumAct);
cnt=1;
for i = 1:NumAct  
    action = TargetSet((i-1)*3+1:i*3);
    action_dir = strcat(file_dir,action,'/');
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
        
        front = resize_feature2(front,fix_size_front);
        side  = resize_feature2(side,fix_size_side);
        top   = resize_feature2(top,fix_size_top);   

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

% train_index = [2 4 6 8 10];       %% Conventional tests
train_index = [1 3 5 7 9];
% train_index = [1 2 3 4 5 6 7 8 9];  %% Leave-One-Subject-Out Test
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
Dim = size(F_train,2);  % AS1:24 AS2:24 AS3:7 for 1 3 5 7 9 as training
Dim2 = size(F_train,1);
                            

%% Original Eigenface Principal Component Analysis used in Chen Paper

% To compute FLOPS Uncomment the commented lines
% FLOPS COMPUTATION

% Inorder to compute the number of FLOPS we are using the FLOPS function 
% which requires us to store the variables produced by the functions in
% workspace, So we are using profile on and profileStruct as follows, refer
% the below link for more details about computing FLOPS
% https://in.mathworks.com/matlabcentral/fileexchange/50608-counting-the-floating-point-operations-flops

tic
%profile on  
disc_set = Eigenface_f(single(F_train),Dim); %% Make sure you save the variable name Eigenface_f at the end of Eigenface_f function
%profileStruct = profile('info');
%[flopTotal,Details]  = FLOPS('Eigenface_f','Eigenface_f',profileStruct);
%profile off

toc
F_train1 = disc_set'*F_train;
F_test1  = disc_set'*F_test;
F_train1 = F_train1./(repmat(sqrt(sum(F_train1.*F_train1)), [Dim,1]));
F_test1  = F_test1./(repmat(sqrt(sum(F_test1.*F_test1)), [Dim,1]));

%% Reduced Basics Decomposition
% Based on the original paper written by Yanlai Chen,"Reduced Basis Decomposition: a Certified and
% Fast Lossy Data Compression Algorithm"
% Paper Link: https://arxiv.org/abs/1503.05947
% Code Link: https://in.mathworks.com/matlabcentral/fileexchange/50125-reduced-basis-decomposition

num=Dim;  %%AS3=103 AS2=122  AS1=120
% Norm1 = ones(Dim2);         %% All ones Matrix Norm(proposed)
% Norm1 = diag(diag(ones(Dim2)));   %% Identity Matrix Norm
% Dim22 = randi(Dim2,Dim2);
% Norm1 = diag((diag(Dim22')));   %% Diagonal Matrix Norm
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


% To compute FLOPS Uncomment the commented lines
% FLOPS COMPUTATION

% Inorder to compute the number of FLOPS we are using the FLOPS function 
% which requires us to store the variables produced by the functions in
% workspace, So we are using profile on and profileStruct as follows, refer
% the below link for more details about computing FLOPS
% https://in.mathworks.com/matlabcentral/fileexchange/50608-counting-the-floating-point-operations-flops

%profile on  
tic
[a, b] = RBD(single(F_train),1e-20,num);   %% Make sure you save the variable name RBD at the end of RBD function
toc
%profileStruct = profile('info');
%[flopTotal,Details]  = FLOPS('RBD','RBD',profileStruct);%
%profile off



% [a, b] = RBD(single(F_train),1e-20,num);  %%AS3  num=103;
% 
% [a, b] = RBD(single(F_train),1e-20,basnum,1,zeros(nr*nc,1));
F_train2 = a'*F_train;
F_test2  = a'*F_test;
% F_train(find(isnan(F_train)))=0;
% F_test(find(isnan(F_test)))=0;

%% This Classifier used in Chen Paper

% Chen, Chen, Kui Liu, and Nasser Kehtarnavaz.
% "Real-time human action recognition based on depth motion maps." 
% Journal of real-time image processing 12, no. 1 (2016): 155-163.

%profile on
tic
label = L2_CRC(F_train1, F_test1, F_train_size, NumAct, lambda);  %% Make sure you save the variable name L2_CRC at the end of L2_CRC function
toc
%profileStruct = profile('info');
%[flopTotal,Details]  = FLOPS('L2_CRC','L2_CRC',profileStruct);%
%profile off
[confusion, accuracy1, CR, FR] = confusion_matrix(label, F_test_size);
fprintf('Accuracy = %f\n', accuracy1);


%%  Classifier used in Our Paper

% Arunraj, Muniandi, Andy Srinivasan, and A. Vimala Juliet. 
% "Online action recognition from RGB-D cameras based on reduced basis 
% decomposition." Journal of Real-Time Image Processing (2018): 1-16.

data.tr_descr  = F_train2;
data.tt_descr  = F_test2;
data.tr_label   = otrainlabel';
data.tt_label  = otestlabel;
dataset.label = tottrainlabel;
% class_num = length(unique(data.tr_label));
 
params.gamma     =  0.01;      %% Set params.class_num = 14,15 in case of 0
% params.gamma     =  0.1; %% Default 1e-2, 0.1-AS3, 0.01-AS2, 0.01-AS1
% params.lambda     =  [1e-1];
% params.lambda     =  0;   %% good accuracy
params.lambda     = 0.1;   %% Default 1e-1, 0.001-AS3, 0.1-AS2, 0.1-AS1
params.class_num  =  20; %% 14 for gamma=0, %% AS1 = 20, AS2 = 30, AS3 = 20 
params.dataset_name = 'MSR-Action3D';

params.model_type =  'ProCRC';


% FLOPS COMPUTATION

% Inorder to compute the number of FLOPS we are using the FLOPS function 
% which requires us to store the variables produced by the functions in
% workspace, So we are using profile on and profileStruct as follows, refer
% the below link for more details about computing FLOPS
% https://in.mathworks.com/matlabcentral/fileexchange/50608-counting-the-floating-point-operations-flops


%profile on
tic
Alpha = ProCRC(data, params);        %% Make sure you save the variable name ProCRC at the end of ProCRC function
toc
%profileStruct = profile('info');
% [flopTotal,Details]  = FLOPS('ProCRC','ProCRC',profileStruct);%
%profile off

%profile on
%tic
[pred_tt_label, ~] = ProMax(Alpha, data, params);    %% Make sure you save the variable name ProMax at the end of ProMax function
%toc
%profileStruct = profile('info');
%[flopTotal,Details]  = FLOPS('ProMax','exampleScriptMAT',profileStruct);%
%profile off

accuracy = (sum(pred_tt_label == data.tt_label)) / length(data.tt_label);
cMat1 = confusionmat(otestlabel,pred_tt_label');

fprintf('Accuracy = %f\n', accuracy);
fileID11 = fopen('../results/FinalResults_AS1_lanczos.txt','w');
fprintf(fileID11,'Actionset1 Lanczos Pro-CRC Results %f\n L2-CRC Results %f \n',(accuracy*100), (accuracy1*100))
% fprintf(['\nThe accuracy on the ', params.dataset_name, ' dataset ',' with gamma=',num2str(params.gamma), ' and lambda=',num2str(params.lambda), ' is ', num2str(roundn(accuracy,-3)),'\n'])
% 
fclose(fileID11)

