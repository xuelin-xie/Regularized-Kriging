function   mse  = MSE(x)
%% 读入Kriging模型
krig1=evalin('base','krig1');
%% 判断维度是否一致
[m n] = size(krig1.S);  % number of design sites and number of dimensions
sx = size(x);            % number of trial sites and their dimension
if  min(sx) == 1 & n > 1 % Single trial point
    nx = max(sx);
    if  nx == n
        mx = 1;  x = x(:).';
    end
    
else
    mx = sx(1);  nx = sx(2);
end
if  nx ~= n
    error(sprintf('Dimension of trial sites should be %d',n))
end
  
  %% 标准化数据
  x = (x - repmat(krig1.Ssc(1,:),mx,1)) ./ repmat(krig1.Ssc(2,:),mx,1);
  dx = repmat(x,m,1) - krig1.S;  % distances to design sites
  f  = feval(@regpoly1, x);
  r  = feval(@corrgauss, krig1.theta, dx);
  
  %% 均方误差的获取
  rt = krig1.C \ r;
  u = krig1.Ft.' * rt - f.';
  v = krig1.G \ u;
  mse =repmat(krig1.sigma2,mx,1) .* repmat((1 + sum(v.^2) - sum(rt.^2))',1,1);            
end

