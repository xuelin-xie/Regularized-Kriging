function   CVmse  = EPeTKfold(S,Y,k,alpha,gamma)
%%  K-fold cross validation
[~,dim]=size(S);
theta = 10; 
lob = 1e-2;
upb = 1e+2;   
theta = repmat(theta, 1, dim) ;
lob = repmat(lob, 1, dim);
upb =  repmat(upb, 1, dim);
[M,N] = size(S);
indices = crossvalind('Kfold',size(S,1),k);
CVmse=0;

for i=1:k
test = (indices == i); 
train = ~test;
train_S = S(train,:);
train_Y = Y(train,:);
test_S = S(test,:);
test_Y = Y(test,:);

if isempty([test_S]) == 1
    Error=sprintf('Need to change the initial point\nleast squares problem is underdetermined');
    error(Error)
end

dmodel =epetdacefit(train_S, train_Y, @regpoly1, @corrgauss, theta, lob, upb,alpha,gamma);
pred = predictor(test_S, dmodel);
CVmse1 = sum((test_Y -pred).*(test_Y-pred))/size(pred,1);
CVmse=CVmse+CVmse1;
end 
CVmse=CVmse/k;

end

