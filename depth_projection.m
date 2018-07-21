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
% [X, zeroPixels] = Kinect_DepthNormalization (X)
% [X] = rotImg3( X, 1*pi/4 , [0 1 0 ])
[rows cols D] = size(X);
% frame_remove0 = 0;
frame_remove1 = 5;
frame_remove2 = 5;  %% 6 for AS2,AS3
frame_remove3 = 5;  %% 7 for AS2,AS3
frame_remove4 = 10;  %% 10 for AS2,AS3
frame_remove5 = 15; 
frame_remove6 = 15; 

% if D <= 23
%     X = X(:,:,frame_remove0+1:end-frame_remove0);
%     D = D;
%     X2D = reshape(X, rows*cols, D);
%     
% else 
if D < 40
    X = X(:,:,frame_remove1+1:end-frame_remove1);
    D = D-10;
    X2D = reshape(X, rows*cols, D);    
% elseif D >= 50
%     X = X(:,:,frame_remove1+1:end-frame_remove1);
%     D = D-10;
%     X2D = reshape(X, rows*cols, D);    
% elseif D >= 60
%     X = X(:,:,frame_remove2+1:end-frame_remove2);
%     D = D-10;
%     X2D = reshape(X, rows*cols, D);
% elseif D >= 70
%     X = X(:,:,frame_remove3+1:end-frame_remove3);
%     D = D-10;
%     X2D = reshape(X, rows*cols, D);
elseif D >= 100                     %%AS2,AS3
    X = X(:,:,frame_remove3+1:end-frame_remove3);
    D = D-10;
    X2D = reshape(X, rows*cols, D);
% elseif D >= 200
%     X = X(:,:,frame_remove3+1:end-frame_remove3);
%     D = D-10;
%     X2D = reshape(X, rows*cols, D);
end



% % enhance edge
% for i = 1:D
%     tmp = X(:,:,i);
%     E = edge(tmp,'canny');
%     tmp(E) = max(tmp(:));
%     X(:,:,i) = tmp;   
% end


% Tensor Decomposition Need review
% tic
% for i = 1:D
%     tmp = X(:,:,i);
%     [ A, B, C ] = svd(tmp);
%     [ tensorfull ] = A*B*C.';
%      X(:,:,i) = tensorfull;   
% end
% toc

% SVD Decomposition Code  Very Slow
% tic
% for i = 1:D
%     tmp = X(:,:,i);
%    compressed = svd_compression(tmp);
%    Irec = svd_decompression(compressed);
%      X(:,:,i) = Irec;   
% end
% toc



% X2D = reshape(X, rows*cols, D);

max_depth = max(X2D(:));



F = zeros(rows, cols);
S = zeros(rows, max_depth);
T = zeros(max_depth, cols);

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
        F = F + abs(front - front_pre);
        S = S + abs(side - side_pre);
        T = T + abs(top - top_pre);
    end   
    
    front_pre = front;
    side_pre  = side;
    top_pre   = top;
    
end
% F = F + edge(abs(front - front_pre),'sobel');
% S = S + edge(abs(side - side_pre),'sobel');
% T = T + edge(abs(top - top_pre),'sobel');



 F = bounding_box(F);
 S = bounding_box(S);
 T = bounding_box(T);
%  save depthprojection



