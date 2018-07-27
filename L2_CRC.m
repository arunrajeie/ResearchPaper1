function label = L2_CRC(F_train, F_test, F_train_size, NumC, lambda)

% NumC:  number of classes         %% Represent the Nc part of Big O
% F_train_size:  number of training samples for each class (1 x NumC)
% F_train_size = [14,6,6,6,6,6,6,6];   %% represent the r part of Big O

error = zeros(1,NumC);
label = zeros(1,size(F_test,2));
dim = size(F_train,2);

for i = 1:size(F_test,2)
    test = F_test(:,i);
    norms = F_train - repmat(test,[1 dim]);
    norms = sum(norms.^2);
    G = lambda .* diag(norms);    
    weights = (F_train'*F_train + G) \ (F_train'*test);

    start = 0;
    for j = 1:NumC
        F = F_train(:,start+1:F_train_size(j)+start);
        w = weights(start+1:F_train_size(j)+start);
        test_hat = F * w;
        error(j) = sum((test-test_hat).^2);
        start = start + F_train_size(j);        
    end
    k = find(error==min(error));
    label(i) = k(1);
end 
%Avoid storing variables during normal operation
save ('../results/L2_CRC.mat')  %% We are saving this variable to count the FLOPS
