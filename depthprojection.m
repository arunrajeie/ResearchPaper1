file_dir = 'MSR-Action3D\';
ActionNum = ['a02', 'a03', 'a05', 'a06', 'a10', 'a13', 'a18', 'a20'; % first row corresponds to action subset 'AS1'
             'a01', 'a04', 'a07', 'a08', 'a09', 'a11', 'a14', 'a12'; % second row corresponds to action subset 'AS2'
             'a06', 'a14', 'a15', 'a16', 'a17', 'a18', 'a19', 'a20']; % third row corresponds to action subset 'AS3'
            
NumAct = 8;          % number of actions in each subset
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

for i = 1:NumAct  
    action = TargetSet((i-1)*3+1:i*3);
    action_dir = strcat(file_dir,action,'\');
    fpath = fullfile(action_dir, '*.mat');
    depth_dir = dir(fpath);
    ind = zeros(1,length(depth_dir));
    flopTotal = zeros(1,length(depth_dir)); 
    for j = 1:length(depth_dir)
        depth_name = depth_dir(j).name;
        sub_num = str2double(depth_name(6:7));
        ind(j) = sub_num;
        load(strcat(action_dir,depth_name));
        depth = depth(:,:,frame_remove+1:end-frame_remove);
%         depth = depth_preprocess(depth);
%         depth = depth_normalize(depth);
%         profile on
        [front, side, top] = depth_projection(depth);   
%         profileStruct = profile('info');
%         [flopTotal(j),Details]  = FLOPS('depth_projection','exampleScriptMAT',profileStruct);%

        profile on
        front = resize_feature1(front,fix_size_front);
        profileStruct = profile('info');
        [flopTotal(j),Details]  = FLOPS('resize_feature1','exampleScriptMAT',profileStruct);%
%         side  = resize_feature1(side,fix_size_side);
%         top   = resize_feature1(top,fix_size_top);   
% 
%         TotalFeature(:,sum(OneActionSample)+j) = [front;side;top];
%         strval=[action(2:3) depth_name(6:7) depth_name(6:7)];
%         tottrainlabel(cnt,1)=str2double(action(2:3));
        cnt=cnt+1;
        
        end
    OneActionSample(i) = length(depth_dir);
    subject_ind{i} = ind;
        
end

% TotalFeature = TotalFeature(:,1:sum(OneActionSample));