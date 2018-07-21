function x = resize_feature1(y,fix_size)

x = imresize(y, [fix_size(1) fix_size(2)], 'bicubic');
% x = imresize(y, [fix_size(1) fix_size(2)], 'lanczos3');
x = x(:);
mask = x;
% mask = adapthisteq(x);   %% need better function
if min(x) < 0
    x = x + abs(min(x)) * 2;
end
x(mask==0) = 0;
% x = imreconstruct(x,mask);
x = (x-min(x)) ./ (max(x)-min(x));
% save exampleScriptMAT