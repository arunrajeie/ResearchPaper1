% Human action recognition
% Test dataset : MSR Action3D
% Cross Subject Test
% by Chen Chen, The University of Texas at Dallas
% chenchen870713@gmail.com
%% Some Part of the code is acquired from the above author, while the major modification
%% is done by the Author of this publication arunrajeie@gmail.com.

file_dir = '../data/MSR-Action3D/';

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
        %for z = 1:PCs,                                                         
        %Reduced_V(:,1) =[]; 
        %end 
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
        % For printing a particular actions and its trials, please manipulate the value of
        % i(Actions1-20) and j(Subjects10 and its trials) in the below loops
        
        if (i == 1) && (j == 1)   %% Change the value to save the corresponding actions of a particular subjects
        front=front-min(front(:)); % shift data such that the smallest element of A is 0
        front=front/max(front(:)); % normalize the shifted data to 1             
        front1=front1-min(front1(:));
        front1=front1/max(front1(:)); 
        front11=front11-min(front11(:)); 
        front11=front11/max(front11(:));  
        front22=front22-min(front22(:)); 
        front22=front22/max(front22(:));  
        front33=front33-min(front33(:)); 
        front33=front33/max(front33(:));  
        front2=front2-min(front2(:)); 
        front2=front2/max(front2(:)); 
        front3=front3-min(front3(:)); 
        front3=front3/max(front3(:));
        front4=front4-min(front4(:)); 
        front4=front4/max(front4(:)); 
        imwrite(front, '../results/Front-OriginalImage.jpg');
        imwrite(front1, '../results/FrontRBD-AllOnes.jpg');
        imwrite(front11, '../results/FrontRBD-Identity.jpg');
        imwrite(front22, '../results/FrontRBD-Diagonal.jpg');
        imwrite(front33, '../results/FrontRBD-SPD.jpg');
        imwrite(front2, '../results/PCA-ReconstructedImage.jpg');
        imwrite(front3, '../results/SVD-ReconstructedImage.jpg');
        imwrite(front4,'../results/SVD-ReconstructedImage.jpg');  %% using svds not included in paper
        end


        front = resize_feature1(front,fix_size_front);
        side  = resize_feature1(side,fix_size_side);
        top   = resize_feature1(top,fix_size_top);   
%         Des   = resize_feature1(Des,fix_size_temp1); 

%         TotalFeature(:,sum(OneActionSample)+j) = [front;side;top;Des];

       
       
        TotalFeature(:,sum(OneActionSample)+j) = [front;side;top];
        
        strval=[action(2:3) depth_name(6:7) depth_name(6:7)];
        tottrainlabel(cnt,1)=str2double(action(2:3));
        cnt=cnt+1;
        break;  %% To break the loop after printing the images, can be removed if you want the images from an entire actionset
    end
    OneActionSample(i) = length(depth_dir);
    subject_ind{i} = ind;
    break;   %% To break the loop after printing the images, can be removed if you want the images for all actions
    
end
% profileStruct = profile('info');
% [flopTotal,Details]  = FLOPS('depth_projection1','depthprojection1',profileStruct);%
TotalFeature = TotalFeature(:,1:sum(OneActionSample));


% save(strcat(ActionSet,'.Features.mat'), 'TotalFeature');
% save('../data/Totalfeaturefull1');





