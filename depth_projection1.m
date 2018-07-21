function [F,S,T] = depth_projection1(X)

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
X2D = reshape(X, rows*cols, D);
max_depth = max(X2D(:));

F = zeros(rows, cols);
S = zeros(rows, max_depth);
T = zeros(max_depth, cols);

for k = 1:D   
    
%     [pcloud1, distance1] = depthToCloud(X(:,:,k));
%     [pcloud2, distance2] = depthToCloud(X(:,:,k+1));
%     ptCloudA = pointCloud(pcloud1);
%     ptCloudB = pointCloud(pcloud2);
%     [ptCloudOut1,indices]= removeInvalidPoints(ptCloudA);
%     [ptCloudOut2,indices]= removeInvalidPoints(ptCloudB);
%     
%     pcshowpair(ptCloudOut1,ptCloudOut2);

      
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
    
%     figure,imshow(front(:,:,1),[]),title('front projection');
%     figure,imshow(side(:,:,1),[]),title('side projection');
%     figure,imshow(top(:,:,1),[]),title('front projection');
    

    
    if k > 1
        
       
        F = F + abs(front - front_pre);
        S = S + abs(side - side_pre);
        T = T + abs(top - top_pre);

    end   
    
  
    front_pre = front;
    side_pre  = side;
    top_pre   = top;
    
end



F = bounding_box(F);
S = bounding_box(S);
T = bounding_box(T);
% [ll lh hl hh] = dwt2(bounding_box(F),'db1');
% F=ll;
% clear ll;
% [ll lh hl hh] = dwt2(bounding_box(S),'db1');
% S=ll;
% clear ll;
% [ll lh hl hh] = dwt2(bounding_box(T),'db1');
% T=ll;
% clear ll;
% save depthprojection1




