function [CVmse,bestalpha,bestgamma] = EPTKGridSearch(S,Y,k,alphamin,alphamax,alphastep,gammamin,gammamax,gammastep)

%% about the Kriging parameters of alpha and gamma
t = 1;
if nargin < 8
    alphastep = 0.05*t;
    gammastep = t;  
end
if nargin < 6
    alphamax = 1;
    alphamin = 0;
end
if nargin < 4
    gammamax = 20;
    gammamin = 1;

end
if nargin < 3
    k = 5;
end

[alpha,gamma] = meshgrid(alphamin:alphastep:alphamax,gammamin:gammastep:gammamax);
[m,n]= size(alpha);
lmmse = zeros(m,n);

bestalpha = 0;
bestgamma = 0;
CVmse = Inf;
for i = 1:m
    for j = 1:n
        lmmse(i,j) = EPeTKfold(S,Y,k,alpha(i,j),10^(1/2*(-11+gamma(i,j))));
        if lmmse(i,j) < CVmse
            CVmse = lmmse(i,j);
            bestalpha = alpha(i,j);
            bestgamma = 10^(1/2*(-11+gamma(i,j)));
        end        
    end
end

%% [lmmse,ps] = mapminmax(lmmse,0,1);
% figure;
% [C,h] = contour(gamma,alpha,lmmse);
% clabel(C,h,'FontSize',16,'Color','r');
% xlabel('gamma','FontSize',18);
% ylabel('alpha','FontSize',18);
% firstline = 'Contour diagram of PBLK parameters'; 
% secondline = ['Best gamma=',num2str(bestgamma),' alpha=',num2str(bestalpha), ...
%     ' CVmse=',num2str(mse)];
% title({firstline;secondline},'Fontsize',22);
% grid on;

% figure;
% % meshc(X,Y,lmmse);
% % mesh(alpha,gamma,lmmse);
% surf(alpha,gamma,lmmse);
% % axis([gammamin,gammamax,alphamin,alphamax]);
% xlabel('alpha','FontSize',18);
% ylabel('gamma','FontSize',18);
% zlabel('CV-MSE','FontSize',18);
% firstline = 'Parameter selection of the EPTK model'; 
% secondline = ['Best alpha=',num2str(bestalpha),' Best gamma=',num2str(bestgamma), ...
%     ' CVmse=',num2str(CVmse)];
% title({firstline;secondline},'Fontsize',20);







