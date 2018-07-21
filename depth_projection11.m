function [F,S,T] = depth_projection(X)

% X: depth map (3D)

%       y
%       |   
%       |  
%       |_ _ _ x
%      /
%     /
%    z
%

[rows cols D] = size(X);
% X2D = shiftdim(X, rows*cols);
% X2D = fftshift(X, rows*cols);
% X2D = circshift(X, rows*cols, D);

frame_remove1 = 5;
frame_remove2 = 6;  %% 6 for AS2,AS3
frame_remove3 = 7;  %% 7 for AS2,AS3
frame_remove4 = 10;  %% 10 for AS2,AS3

if D < 50
    X = X(:,:,frame_remove1+1:end-frame_remove1);
    D = D-10;
    X2D = reshape(X, rows*cols, D);
    
elseif D >= 50
    X = X(:,:,frame_remove1+1:end-frame_remove1);
    D = D-10;
    X2D = reshape(X, rows*cols, D);    
elseif D >= 60
    X = X(:,:,frame_remove2+1:end-frame_remove2);
    D = D-12;
    X2D = reshape(X, rows*cols, D);
elseif D >= 70
    X = X(:,:,frame_remove3+1:end-frame_remove3);
    D = D-14;
    X2D = reshape(X, rows*cols, D);
elseif D >= 100                     %%AS2,AS3
    X = X(:,:,frame_remove4+1:end-frame_remove4);
    D = D-20;
    X2D = reshape(X, rows*cols, D);
end

% X2D = reshape(X, rows*cols, D);
max_depth = max(X2D(:));

F = zeros(rows, cols);
S = zeros(rows, max_depth);
T = zeros(max_depth, cols);

% F1 = zeros(rows, cols);
% S1 = zeros(max_depth,rows);
% T1 = zeros(cols,max_depth);

for k = 1:D   
    front = X(:,:,k);
    side = zeros(rows, max_depth);
    top = zeros(max_depth, cols);
    for i = 1:rows
        for j = 1:cols
            if front(i,j) ~= 0
                side(i,front(i,j)) = j;   % side view projection (y-z projection)
                top(front(i,j),j)  = i;   % top view projection  (x-z projection)
            end
        end
    end
    
    if k > 1
        
%         Original
%              
        F = F + abs(front - front_pre);
        S = S + abs(side - side_pre);
        T = T + abs(top - top_pre);
 
%         Fast Filter
%          w9 = [0 -1 0; -1 5 -1; 0 -1 0;]; % K11
%          F = F + imfilter(abs(front - front_pre),w9, 'symmetric', 'conv');
%          S = S + imfilter(abs(side - side_pre),w9, 'symmetric', 'conv');
%          T = T+ imfilter(abs(top - top_pre),w9, 'symmetric', 'conv');
        

%         Original Equivalent
%         F = F + imabsdiff(front,front_pre);
%         S = S + imabsdiff(side,side_pre);
%         T = T + imabsdiff(top,top_pre);


        % Absolute Transformed Distance
%         f1 = fwht(front) - fwht(front_pre);
%         f2 = fwht(side) - fwht(side_pre);
%         f3 = fwht(top) - fwht(top_pre);    
       
%         F = fwht(F) + f1;
%         S = fwht(S) + f2;
%         T = fwht(T) + f3;
        
    end   
    front_pre = front;
    side_pre  = side;
    top_pre   = top;
end

% for k = 1:5:D
% for k = 1:D   
%     front1 = X(:,:,k);
%     side1 = zeros(max_depth,rows);
%     top1 = zeros(cols,max_depth);
%     for i = 1:rows
%         for j = 1:cols
%             if front1(i,j) ~= 0
%                 side1(front1(i,j),i) = j;   % side view projection (y-z projection)
%                 top1(j,front1(i,j))  = i;   % top view projection  (x-z projection)
%             end
%         end
%     end
%     
%      if k > 1
%          F1 = F1 + imsharpness(abs(front1 - front_pre1));
%           S1 = S1 +imsharpness(abs(side1 - side_pre1));
%           T1 = T1+ imsharpness(abs(top1 - top_pre1));
%           
%       end   
%     front_pre1 = front1;
%     side_pre1  = side1;
%     top_pre1   = top1;
% end
%     
% F = F-F1;
% S = S-S1';
% T = T-T1';



% F = F + edge(abs(front - front_pre),'sobel');
% S = S + edge(abs(side - side_pre),'sobel');
% T = T + edge(abs(top - top_pre),'sobel');
% F = imfill(F,'holes');
% S = imfill(S,'holes');
% T = imfill(T,'holes');

F = bounding_box(F);
S = bounding_box(S);
T = bounding_box(T);
% save exampleScriptMAT





