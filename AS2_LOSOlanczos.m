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

ActionSet = 'AS2';
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

% load ('../data/Totalfeaturefull');
fid = fopen('../results/AS2LOSO_lanczosresults.txt','w') ;   %% For storing the results

train_total = [1 2 3 4 5 6 7 8 9 10];

for l1 = 1:10
    train_total = [1 2 3 4 5 6 7 8 9 10];
    train_total(l1) = [];
    train_index1(l1,:) = train_total;
end

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


                        
%% Reduced Basics Decomposition with different Error-Estimation Norms

Dim = size(F_train,2);  
Dim1 = size(F_train,1);     

num=Dim1;  %%AS3=103 AS2=122  AS1=120
Norm1 = diag(diag(ones(Dim1)));   %% Identity Matrix Norm
Norm2 = ones(Dim1);         %% All ones Matrix Norm
Dim22 = randi(Dim1,Dim1);
Norm3 = diag((diag(Dim22)));   %% Diagonal Matrix Norm
Norm4 = generateSPDmatrix(Dim1);  %% SPD Matrix Norm1 


[a, b] = RBD(single(F_train),1e-20,num,Norm1);  
F_train1 = a'*F_train;
F_test1  = a'*F_test;
% F_train(find(isnan(F_train)))=0;  %% Optional
% F_test(find(isnan(F_test)))=0;

[a1, b1] = RBD(single(F_train),1e-20,num,Norm2);  
F_train2 = a1'*F_train;
F_test2  = a1'*F_test;

[a2, b2] = RBD(single(F_train),1e-20,num,Norm3);  
F_train3 = a2'*F_train;
F_test3  = a2'*F_test;

[a3, b3] = RBD(single(F_train),1e-20,num,Norm4);  
F_train4 = a3'*F_train;
F_test4  = a3'*F_test;


%% Testing

%////////////////////////////////////////////////////////////////////%    
%         Tikhonov regularized Collaborative Classifier              %
%////////////////////////////////////////////////////////////////////%

label1 = L2_CRC(F_train1, F_test1, F_train_size, NumAct, lambda);
[confusion, accuracy1(row1), CR, FR] = confusion_matrix(label1, F_test_size);
fprintf(fid,'Accuracy of L2CRC with RBD-Identity Norm = %f\n', accuracy1(row1));

label2 = L2_CRC(F_train2, F_test2, F_train_size, NumAct, lambda);
[confusion, accuracy2(row1), CR, FR] = confusion_matrix(label2, F_test_size);
fprintf(fid,'Accuracy of L2CRC with RBD-All ones Norm(Proposed) = %f\n', accuracy2(row1));

label3 = L2_CRC(F_train3, F_test3, F_train_size, NumAct, lambda);
[confusion, accuracy3(row1), CR, FR] = confusion_matrix(label3, F_test_size);
fprintf(fid,'Accuracy of L2CRC with RBD-Diagonal Norm = %f\n', accuracy3(row1));

label4 = L2_CRC(F_train4, F_test4, F_train_size, NumAct, lambda);
[confusion, accuracy4(row1), CR, FR] = confusion_matrix(label4, F_test_size);
fprintf(fid,'Accuracy of L2CRC with RBD-SPD Norm = %f\n', accuracy4(row1));

%%  Probability Based Collaborative Representation Classifier (Identiy Norm)

data.tr_descr  = F_train1;
data.tt_descr  = F_test1;
data.tr_label   = otrainlabel';
data.tt_label  = otestlabel;
dataset.label = tottrainlabel;

%% Make sure to use the gamma, lambda values given in the paper to reproduce the results
params.gamma     =  [1e-2];  
% params.lambda     =  [1e-1];
params.lambda     =  [1e-1];   %% good accuracy
params.class_num  =  20; %% 14 for gamma=0, %% AS1 = 20, AS2 = 30, AS3 = 20 (try different values)
% params.class_num  =  30; 
params.dataset_name = 'MSR-Action3D';
params.model_type =  'ProCRC';

Alpha = ProCRC(data, params);
[pred_tt_label, ~] = ProMax(Alpha, data, params);


accuracy11(row1) = (sum(pred_tt_label == data.tt_label)) / length(data.tt_label);
cMat1 = confusionmat(otestlabel,pred_tt_label');

fprintf(fid,['\nThe accuracy of ProCRC with RBD-Identity Norm on the ', params.dataset_name, ' dataset ',' with gamma=',num2str(params.gamma), ' and lambda=',num2str(params.lambda), ' is ', num2str(roundn(accuracy11(row1),-3)),'\n'])
fprintf(fid,'\n') ;
clear data;

%%  Probability Based Collaborative Representation Classifier (Allones Norm)
data.tr_descr  = F_train2;
data.tt_descr  = F_test2;
data.tr_label   = otrainlabel';
data.tt_label  = otestlabel;
dataset.label = tottrainlabel;

%% Make sure to use the gamma, lambda values given in the paper to reproduce the results
params.gamma     =  [1e-2];
% params.lambda     =  [1e-1];
params.lambda     =  [1];   %% good accuracy
params.class_num  =  30; %% 14 for gamma=0, %% AS1 = 20, AS2 = 30, AS3 = 20 (try different values)
% params.class_num  =  30; 
params.dataset_name = 'MSR-Action3D';
params.model_type =  'ProCRC';

Alpha = ProCRC(data, params);
[pred_tt_label, ~] = ProMax(Alpha, data, params);


accuracy22(row1) = (sum(pred_tt_label == data.tt_label)) / length(data.tt_label);
cMat1 = confusionmat(otestlabel,pred_tt_label');

fprintf(fid,['\nThe accuracy of ProCRC with RBD-All ones Norm(Proposed) ', params.dataset_name, ' dataset ',' with gamma=',num2str(params.gamma), ' and lambda=',num2str(params.lambda), ' is ', num2str(roundn(accuracy22(row1),-3)),'\n'])
fprintf(fid,'\n') ;
clear data;

%%  Probability Based Collaborative Representation Classifier (Diagonal Norm)
data.tr_descr  = F_train3;
data.tt_descr  = F_test3;
data.tr_label   = otrainlabel';
data.tt_label  = otestlabel;
dataset.label = tottrainlabel;

%% Make sure to use the gamma, lambda values given in the paper to reproduce the results
params.gamma     =  [1e-2];
% params.lambda     =  [1e-1];
params.lambda     =  [1e-2];   %% good accuracy
params.class_num  =  30; %% 14 for gamma=0, %% AS1 = 20, AS2 = 30, AS3 = 20 (try different values)
% params.class_num  =  30; 
params.dataset_name = 'MSR-Action3D';
params.model_type =  'ProCRC';

Alpha = ProCRC(data, params);
[pred_tt_label, ~] = ProMax(Alpha, data, params);


accuracy33(row1) = (sum(pred_tt_label == data.tt_label)) / length(data.tt_label);
cMat1 = confusionmat(otestlabel,pred_tt_label');

fprintf(fid,['\nThe accuracy of ProCRC with RBD-Diagonal Norm ', params.dataset_name, ' dataset ',' with gamma=',num2str(params.gamma), ' and lambda=',num2str(params.lambda), ' is ', num2str(roundn(accuracy33(row1),-3)),'\n'])
fprintf(fid,'\n') ;
clear data;

%%  Probability Based Collaborative Representation Classifier (SPD Norm)
data.tr_descr  = F_train4;
data.tt_descr  = F_test4;
data.tr_label   = otrainlabel';
data.tt_label  = otestlabel;
dataset.label = tottrainlabel;

%% Make sure to use the gamma, lambda values given in the paper to reproduce the results
params.gamma     =  [1e-2];
% params.lambda     =  [1e-1];
params.lambda     =  [1e-2];   %% good accuracy
params.class_num  =  30; %% 14 for gamma=0, %% AS1 = 20, AS2 = 30, AS3 = 20 (try different values)
% params.class_num  =  30; 
params.dataset_name = 'MSR-Action3D';
params.model_type =  'ProCRC';

Alpha = ProCRC(data, params);
[pred_tt_label, ~] = ProMax(Alpha, data, params);


accuracy44(row1) = (sum(pred_tt_label == data.tt_label)) / length(data.tt_label);
cMat1 = confusionmat(otestlabel,pred_tt_label');

fprintf(fid,['\nThe accuracy of ProCRC with RBD-SPD Norm ', params.dataset_name, ' dataset ',' with gamma=',num2str(params.gamma), ' and lambda=',num2str(params.lambda), ' is ', num2str(roundn(accuracy44(row1),-3)),'\n'])
fprintf(fid,'\n') ;

clear train_index;


end
fprintf(fid,'Average accuracy of LOSO Lanczos L2CRC with RBD(Identity Norm) = %f; and Standard Deviation = %f \n', mean(accuracy1), std(accuracy1));
fprintf(fid,'Average accuracy of LOSO Lanczos L2CRC with RBD(Allones Norm - Proposed) = %f; and Standard Deviation = %f \n', mean(accuracy2), std(accuracy2));
fprintf(fid,'Average accuracy of LOSO Lanczos L2CRC with RBD(Digonal Norm) = %f; and Standard Deviation = %f \n', mean(accuracy3), std(accuracy3));
fprintf(fid,'Average accuracy of LOSO Lanczos L2CRC with RBD(SPD Norm) = %f; and Standard Deviation = %f \n', mean(accuracy4), std(accuracy4));

fprintf(fid,'Average accuracy of LOSO Lanczos Pro-CRC with RBD(Identity Norm) = %f; and Standard Deviation = %f \n', mean(accuracy11), std(accuracy11));
fprintf(fid,'Average accuracy of LOSO Lanczos Pro-CRC with RBD(Allones Norm - Proposed) = %f; and Standard Deviation = %f \n', mean(accuracy22), std(accuracy22));
fprintf(fid,'Average accuracy of LOSO Lanczos Pro-CRC with RBD(Digonal Norm) = %f; and Standard Deviation = %f \n', mean(accuracy33), std(accuracy33));
fprintf(fid,'Average accuracy of LOSO Lanczos Pro-CRC with RBD(SPD Norm) = %f; and Standard Deviation = %f \n', mean(accuracy44), std(accuracy44));

fclose(fid) ;
