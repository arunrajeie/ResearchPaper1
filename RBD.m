function [bases, TransMat] = RBD(data, tol, col, NormMat, StartingVec)
%% Reduced basis decomposition
% This function approximate a matrix (data) by a product of two matrices
% (bases and TransMat) with the column space of the first matrix
% approximating that of the input matrix.
%
% There are multiple ways to call it as you can specify from 1 to 5 inputs
%
% RBD(data), RBD(data,tol), RBD(data,tol,col) etc

% Inputs:
%
% Provide only the first three unless you understand what you are doing
%
% * data => the data matrix you want to decompose
% * tol  => the accuracy you desire of your decomposition
% * col  => the number of columns you can afford to have in the compressed
%           matrix
% * NormMat => A SPD matrix if you wish to use a different norm to measure
%           the error
% * StartingVec => The vector you wish to use to start the greedy algorithm
%
% Outputs: 
%
% bases*TransMat $\approx$ data
%
% * bases    => the "compressed" matrix (to be more accurate, the group of vectors spanning
% approximately the column space of data)
% * TransMat => the "transformation" matrix
%
% Author: Yanlai Chen
%
% Version: 1.0
%
% Date: 03/20/2015

%% Error Checking
% We provide values to those inputs that are not supplied

if(nargin<1)
    error('RBD needs at least one input, a matrix')
end

if (nargin == 1)
    tol = 1e-6;
    col = min(10,size(data,2));
    TrueE_NoR = 1;
    StartingVec = 0;
elseif(nargin == 2)
    col = size(data,2);
    TrueE_NoR = 1;
    StartingVec = 0;
elseif(nargin == 3)
    TrueE_NoR = 1;
    StartingVec = 0;
elseif(nargin == 4)
    TrueE_NoR = 0;
    StartingVec = 0;
else
    TrueE_NoR = 0;
end

%% Praparation of the algorithm

nr = size(data,1);
nc = size(data,2);

bases = zeros(nr,col);

TransMat = zeros(col,nc);
xiFlag=zeros(1,col);
xiFlag(1) = randi(nc);
i=1;
CurErr = tol + 1;

% Preparation for efficient error evaluation
ftf = sum(data'.*data',2);
AtAAtXi = zeros(nc,col);

if(TrueE_NoR == 0)
    AtAAt = data'*NormMat;
    XitAAtXi = zeros(col,col);
    tM = data'*NormMat;
% a very fast way to evaluate ftf(j) = data(:,j)'*AAt*data(:,j);
    ftf = sum(tM.*data',2);
end

%% The RBD greedy algorithm

while (i <= col) && (CurErr > tol)
    
    if((i==1) && norm(StartingVec) > 1e-6)
        biCand = StartingVec;
    else
        biCand = data(:,xiFlag(i));
    end
%% Inside: Gram-Schmidt orthonormalization of the current candidate with all
% previsouly chosen basis vectors


    for j=1:i-1
        biCand = biCand - (biCand'*bases(:,j))*bases(:,j);
    end
    normi = sqrt(biCand'*biCand);
    if(normi < 1e-7)
        fprintf('Reduced system getting singular - to stop with %d basis functions\n',i-1);
        bases = bases(:,1:i-1);
        TransMat = TransMat(1:i-1,:);
        break
    else
        bases(:,i) = biCand/normi;
    end
    TransMat(i,:) = bases(:,i)'*data;
    
%% Inside: With one more basis added, we need to update what allows for the
% efficient error evaluation.
    if(TrueE_NoR == 0)
        AtAAtXi(:,i) = AtAAt*bases(:,i);
        XitAAtXi(i,1:i) = bases(:,i)'*NormMat*bases(:,1:i);
        XitAAtXi(1:i,i) = XitAAtXi(i,1:i)';
    else
        AtAAtXi(:,i) = data'*bases(:,i);
    end
%% Inside: Efficiently go through all the columns to identify where the error would
% be the largest if we were to use the current space for compression.
    TMM = TransMat(1:i,:);
    te1 = sum(AtAAtXi(:,1:i).*TMM',2);
    if(TrueE_NoR == 0)
        tM = TMM'*XitAAtXi(1:i,1:i);
    else
        tM = TMM';
    end
    te2 = sum(tM.*TMM',2);
    errord = ftf - 2*te1 + te2;
    [CurErr,TempPos] = max(errord);

% Mark this location for the next round
    if(i < col)
        xiFlag(i+1) = TempPos;
    end
% If the largest error is small enough, we announce and stop.
    if(CurErr <= tol)
        fprintf('Reduced system getting accurate enough - to stop with %d basis functions\n',i);
        bases = bases(:,1:i);
        TransMat = TransMat(1:i,:);
    else
        i = i+1;
    end
end
%Avoid storing variables during normal operation
save ('../results/RBD.mat')  %% We are saving this variable to count the FLOPS
end
