% Human action recognition
% Test dataset : MSR Action3D
% Cross Subject Test
% by Chen Chen, The University of Texas at Dallas
% chenchen870713@gmail.com
%% Some Part of the code is acquired from the above author, while the major modification
%% is done by the Author of this publication arunrajeie@gmail.com.


file_dir = 'MSR-Action3D\';

ActionNum = ['a01','a02','a03','a04','a05','a06','a07','a08','a09','a10','a11','a12','a13'...
    'a14','a15','a16','a17','a18','a19','a20'];
         
% ActionNum = ['a02', 'a03', 'a05', 'a06', 'a10', 'a13', 'a18', 'a20'; % first row corresponds to action subset 'AS1'
%              'a01', 'a04', 'a07', 'a08', 'a09', 'a11', 'a14', 'a12'; % second row corresponds to action subset 'AS2'
%              'a06', 'a14', 'a15', 'a16', 'a17', 'a18', 'a19', 'a20']; % third row corresponds to action subset 'AS3'
            
NumAct = 20;          % 8, 20 number of actions in each subset
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
%                 fix_size_front = [100;50]; fix_size_side = [100;82]; fix_size_top = [82;47]; 
%                 fix_size_front = round([100;50]/2); fix_size_side = round([100;82]/2); fix_size_top = round([82;47]/2);
                fix_size_front = [102,54]; fix_size_side = [102,75]; fix_size_top = [75,54];      

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
        
        %% Image Reconstruction Steps
        
        Dim1 = size(front,2);
        Dim2 = size(front,1);
        Norm11 = diag(diag(ones(Dim2)));   %% Identity Matrix Norm(Default is Identity Matrix so not used in the below steps)
        Norm1 = ones(Dim2);         %% All ones Matrix Norm (Proposed in our paper)        
        Dim22 = randi(Dim2,Dim2);   %% Generating random Diagonal Elements, avoid using normal distribution which produces worst results
        Norm2 = diag((diag(Dim22)));   %% Diagonal Matrix Norm
        Norm3 = generateSPDmatrix(Dim2);  %% SPD Norm 
        

        %% Reduced Basis Decomposition based compression ( with coefficients in the order of 10%,20%,30%,40%,50%,60%,70%,80%,90% as displayed by numbers)
        tic
        [a, b] = RBD(single(front),1e-20,95);       %% 10,19,29,38,48,57,67,76,95
        [a11, b11] = RBD(single(front),1e-20,95,Norm1);       %% 10,19,29,38,48,57,67,76,95
        [a22, b22] = RBD(single(front),1e-20,95,Norm2);       %% 10,19,29,38,48,57,67,76,95
        [a23, b23] = RBD(single(front),1e-20,95,Norm3);       %% 10,19,29,38,48,57,67,76,95
%         [a, b] = RBD(single(front),1e-20,80);
%         [a, b] = RBD(single(front),1e-20,25,Norm1); %% Default 1e-20
        toc
        
        %% Reconstructing the image using reduced basis co-efficients
        
        front1 = a*b;           
        front11 = a11*b11;
        front22 = a22*b22;
        front33 = a23*b23;
       
        %% PCA compression
        
        %%PCA
        Data_mean = mean(front);      
        [a1, b1] = size(front); 
        Data_meanNew = repmat(Data_mean,a1,1); 
        DataAdjust = front - Data_meanNew; 
        cov_data = cov(DataAdjust);   
        [V, D] = eig(cov_data); 
        V_trans = transpose(V); 
        DataAdjust_trans = transpose(DataAdjust);  
        FinalData = V_trans * DataAdjust_trans;   
        
        %%Inverse PCA - Image Reconstruction using all eigenvalues
%         OriginalData_trans = inv(V_trans) * FinalData;                         
%         OriginalData = transpose(OriginalData_trans) + Data_meanNew;  
%         front2 = OriginalData;
        
        %%Compressed PCA
        PCs = Dim1-85;             %% 85,76,66,57,47,38,28,19,0
        PCs = b1 - PCs;                                                         
        Reduced_V = V;                                                         
        for z = 1:PCs,                                                         
        Reduced_V(:,1) =[]; 
        end 
        Y=Reduced_V'* DataAdjust_trans;                                        
        Compressed_Data=Reduced_V*Y;                                           
        Compressed_Data = Compressed_Data' + Data_meanNew;  
        front2 = Compressed_Data;
           
        %% SVD Compression
        
        tic
        [e, f, g] = svd(single(front));
        toc
        front3 = e*f*g';
        
        %% Below is optional SVDS function used in Matlab
        
        tic
        [h, i1, j1] = svds(front,10);       %% 10,19,29,38,48,57,67,76,95
        toc
        front4 = h*i1*j1';
       
        %% Note: 
        % Not much difference b/w RBD obtained via Diagonal norms and SPD(Symmetric Positive Definite Matrix) norms
        % infact both produces similar levels of classification accuracy
        
        figure,imshow(front,[]),title('Front-Original Image');
        figure,imshow(front1,[]),title('FrontRBD-AllOnes');
        figure,imshow(front11,[]),title('FrontRBD-Identity');
        figure,imshow(front22,[]),title('FrontRBD-Diagonal');  
        figure,imshow(front33,[]),title('FrontRBD-SPD');
        figure,imshow(front2,[]),title('PCA-Reconstructed Image');
        figure,imshow(front3,[]),title('SVD-Reconstructed Image');
        figure,imshow(front4,[]),title('SVD2-Reconstructed Image');


        front = resize_feature1(front,fix_size_front);
        side  = resize_feature1(side,fix_size_side);
        top   = resize_feature1(top,fix_size_top);   
%         Des   = resize_feature1(Des,fix_size_temp1); 

%         TotalFeature(:,sum(OneActionSample)+j) = [front;side;top;Des];

       
       
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


% save(strcat(ActionSet,'.Features.mat'), 'TotalFeature');
save('Totalfeaturefull');


%% Generate training and testing data

% load ('TotalfeatureAS1');
% load ('TotalfeatureAS2');
% load ('TotalfeatureAS3');
load('Totalfeaturefull');

% train_index = [2 4 6 8 10];
% train_index = [1 3 5 7 9];
train_index = [1 2 3 4 5 6 7 8 9];
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
profile on  
disc_set = Eigenface_f(single(F_train),Dim); %% Make sure you save the variable name Eigenface_f at the end of Eigenface_f function
profileStruct = profile('info');
[flopTotal,Details]  = FLOPS('Eigenface_f','Eigenface_f',profileStruct);
profile off

toc
F_train = disc_set'*F_train;
F_test  = disc_set'*F_test;
F_train = F_train./(repmat(sqrt(sum(F_train.*F_train)), [Dim,1]));
F_test  = F_test./(repmat(sqrt(sum(F_test.*F_test)), [Dim,1]));

%% Reduced Basics Decomposition
% Based on the original paper written by Yanlai Chen,"Reduced Basis Decomposition: a Certified and
% Fast Lossy Data Compression Algorithm"
% Paper Link: https://arxiv.org/abs/1503.05947
% Code Link: https://in.mathworks.com/matlabcentral/fileexchange/50125-reduced-basis-decomposition

num=Dim;  %%AS3=103 AS2=122  AS1=120
density = 0.01;
% Norm1 = ones(Dim2);         %% All ones Matrix Norm
Norm1 = diag(diag(ones(Dim2)));   %% Identity Matrix Norm
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


% To compute FLOPS Uncomment the commented lines
% FLOPS COMPUTATION

% Inorder to compute the number of FLOPS we are using the FLOPS function 
% which requires us to store the variables produced by the functions in
% workspace, So we are using profile on and profileStruct as follows, refer
% the below link for more details about computing FLOPS
% https://in.mathworks.com/matlabcentral/fileexchange/50608-counting-the-floating-point-operations-flops

profile on  
[a, b] = RBD(single(F_train),1e-20,num,Norm1);   %% Make sure you save the variable name RBD at the end of RBD function
profileStruct = profile('info');
[flopTotal,Details]  = FLOPS('RBD','RBD',profileStruct);%
profile off


toc
% [a, b] = RBD(single(F_train),1e-20,num);  %%AS3  num=103;
% 
% [a, b] = RBD(single(F_train),1e-20,basnum,1,zeros(nr*nc,1));
F_train = a'*F_train;
F_test  = a'*F_test;
% F_train(find(isnan(F_train)))=0;
% F_test(find(isnan(F_test)))=0;

%% Random Projection
% [Dim,Dim1] = size(F_train);
% tic
% disc_set = randn(Dim,Dim1);
% toc
% F_train = disc_set'*F_train;
% F_test  = disc_set'*F_test;

%% Fast PCA
% [m, n] = size(F_train);
% p = min(m,n);
% q = 0;     %% iterations
% s = 0;

% kstep = 200;
% [a, b, c] = rsvd_version1(single(F_train),Dim,p,q,s);    %%% Under Trial
% [a, b, c] = rsvd_version3(single(F_train),Dim,kstep,q,s);    %%% Under Trial
% [a, b, c] = rsvd(single(F_train),Dim);    %%% Under Trial
% F_train = a'*F_train;
% F_test  = a'*F_test;
% F_train = F_train./(repmat(sqrt(sum(F_train.*F_train)), [Dim,1]));
% F_test  = F_test./(repmat(sqrt(sum(F_test.*F_test)), [Dim,1]));

%% The Classifier used in Chen Paper

% Chen, Chen, Kui Liu, and Nasser Kehtarnavaz.
% "Real-time human action recognition based on depth motion maps." 
% Journal of real-time image processing 12, no. 1 (2016): 155-163.

profile on
tic
label = L2_CRC(F_train, F_test, F_train_size, NumAct, lambda);  %% Make sure you save the variable name L2_CRC at the end of L2_CRC function
toc
profileStruct = profile('info');
[flopTotal,Details]  = FLOPS('L2_CRC','L2_CRC',profileStruct);%
profile off
[confusion, accuracy1, CR, FR] = confusion_matrix(label, F_test_size);
fprintf('Accuracy = %f\n', accuracy1);


%%  Classifier used in Our Paper

% Arunraj, Muniandi, Andy Srinivasan, and A. Vimala Juliet. 
% "Online action recognition from RGB-D cameras based on reduced basis 
% decomposition." Journal of Real-Time Image Processing (2018): 1-16.

data.tr_descr  = F_train;
data.tt_descr  = F_test;
data.tr_label   = otrainlabel';
data.tt_label  = otestlabel;
dataset.label = tottrainlabel;
% class_num = length(unique(data.tr_label));
 
params.gamma     =  0.01;      %% Set params.class_num = 14,15 in case of 0
% params.gamma     =  0.1; %% Default 1e-2, 0.1-AS3, 0.01-AS2, 0.01-AS1
% params.lambda     =  [1e-1];
% params.lambda     =  0;   %% good accuracy
params.lambda     = 0.1;   %% Default 1e-1, 0.001-AS3, 0.1-AS2, 0.1-AS1
params.class_num  =  20; %% 14 for gamma=0, %% AS1 = 20, AS2 = 30, AS3 = 20 Represent the number of training samples rmin in Big-O
params.dataset_name = 'MSR-Action3D';

params.model_type =  'ProCRC';


% FLOPS COMPUTATION

% Inorder to compute the number of FLOPS we are using the FLOPS function 
% which requires us to store the variables produced by the functions in
% workspace, So we are using profile on and profileStruct as follows, refer
% the below link for more details about computing FLOPS
% https://in.mathworks.com/matlabcentral/fileexchange/50608-counting-the-floating-point-operations-flops


profile on
tic
Alpha = ProCRC(data, params);        %% Make sure you save the variable name ProCRC at the end of ProCRC function
toc
profileStruct = profile('info');
[flopTotal,Details]  = FLOPS('ProCRC','ProCRC',profileStruct);%
profile off

profile on
tic
[pred_tt_label, ~] = ProMax(Alpha, data, params);    %% Make sure you save the variable name ProMax at the end of ProMax function
toc
profileStruct = profile('info');
[flopTotal,Details]  = FLOPS('ProMax','exampleScriptMAT',profileStruct);%
profile off

accuracy = (sum(pred_tt_label == data.tt_label)) / length(data.tt_label);
cMat1 = confusionmat(otestlabel,pred_tt_label');

fprintf('Accuracy = %f\n', accuracy);
% fprintf(['\nThe accuracy on the ', params.dataset_name, ' dataset ',' with gamma=',num2str(params.gamma), ' and lambda=',num2str(params.lambda), ' is ', num2str(roundn(accuracy,-3)),'\n'])
% 

