function bestmu=OptRPR(S,Y,a0,q,n,k)

%% Optimal regularization parameter
% a0, q, n, k
if nargin < 6
    k = 5;
end
if nargin < 5
    n=20; 
end
if nargin < 4
   q=10^(1/2);
end
if nargin < 3
   a0=10^(-5);
end

for i = 1:n+1
    % mu
    mu = a0*q^(i-1);
    % K折交叉验证
    mse  = RPeTKfold(S,Y,k,mu);
    CVscore(i)= mse;
end

[MinCVscore,j]=min(CVscore);
bestmu = a0*q^(j-1);
disp(['Best mu is ',num2str(bestmu),'.'])

end