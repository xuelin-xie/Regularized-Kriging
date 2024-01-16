
function y=callobj(objfun,x)
x=x';
[~,n]=size(x);
y=[];
for i=1:n
   y0=feval(objfun,x(:,i));
   y=[y;y0];
end