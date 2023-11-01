function krgmodel = buildKRGEPeT(S,Y,bestalpha,bestgamma)
%  Build a surrogate function based on Kriging Functions
[~,dim]=size(S);
%初始化Kriging的相关条件
theta = 10; 
lob = 1e-2;
upb = 1e+2;  
theta = repmat(theta, 1, dim) ;
lob = repmat(lob, 1, dim);
upb =  repmat(upb, 1, dim);
alpha=bestalpha;
gamma = bestgamma;


% 建立Kriging模型
krgmodel =epetdacefit(S,Y,'regpoly1','corrgauss', theta, lob, upb, alpha, gamma);

return
