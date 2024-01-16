function krgmodel = buildKRGRPeT(S,Y,bestlambda)
%  Build a surrogate function based on Kriging Functions
[~,dim]=size(S);
%初始化Kriging的相关条件
theta = 10; 
lob = 1e-2;
upb = 1e+2; 
theta = repmat(theta, 1, dim) ;
lob = repmat(lob, 1, dim);
upb =  repmat(upb, 1, dim);
lambda = bestlambda;


% 建立Kriging模型
krgmodel =ridgepetdacefit(S,Y,'regpoly1','corrgauss', theta, lob, upb, lambda);

return
