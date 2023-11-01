clc;
clear;
setdemorandstream(pi);
problem.f=@copeak; % Test functions // copeak/ Drop/ langermann/ morcaf95a ....
[XL,XU]=copeak_bound();
bounds=[XL;XU];
dim=size(bounds,2); 
pointnum=60;   %60/ 90

for i=1:10
%% Sampling and evaluation points
S=LHD(XL,XU,pointnum); 
Y=callobj(problem.f,S);
EX=LHD(XL,XU,5000);  
EY=callobj(problem.f,EX); 

%% Ordinary Kriging
krig1=buildKRG(S,Y); 
%% predicted values
K= predictor(EX, krig1);
%% The evaluation index of the Kriging model 
R2(i)=1-sum((EY -K).*(EY-K)) /sum((EY-mean(EY)).*(EY-mean(EY)));
RMSE1(i)=sqrt(sum((EY -K).*(EY -K)) /size(EY,1));
MAE1(i)=sum(abs(EY -K))/size(EY,1);

%% Lasso-Kriging
%% Obtain optimal parameters
bestlambda=OptRPL(S,Y);
krig2=buildKRGLPeT(S,Y,bestlambda); 
%% predicted values
LK= predictor(EX, krig2);
%% The evaluation index of the LK model
LR2(i)=1-sum((EY -LK).*(EY-LK)) /sum((EY-mean(EY)).*(EY-mean(EY)));
RMSE2(i)=sqrt(sum((EY -LK).*(EY -LK)) /size(EY,1));
MAE2(i)=sum(abs(EY -LK))/size(EY,1);


%% Ridge-Kriging
%% Obtain optimal parameters
bestmu=OptRPR(S,Y);
krig3=buildKRGRPeT(S,Y,bestmu);   
%% predicted values
RK= predictor(EX, krig3);
%% The evaluation index of the RK model
RR2(i)=1-sum((EY -RK).*(EY-RK)) /sum((EY-mean(EY)).*(EY-mean(EY)));
RMSE3(i)=sqrt(sum((EY -RK).*(EY -RK)) /size(EY,1));
MAE3(i)=sum(abs(EY -RK))/size(EY,1);

%% Elastic-net Kriging
% Obtain optimal parameters
[CVmse,bestalpha,bestgamma] = EPTKGridSearch(S,Y,5);
krig4=buildKRGEPeT(S,Y,bestalpha,bestgamma);
%% predicted values
EK= predictor(EX, krig4);
%% The evaluation index of the EK model
PR2(i)=1-sum((EY -EK).*(EY-EK)) /sum((EY-mean(EY)).*(EY-mean(EY)));
RMSE4(i)=sqrt(sum((EY -EK).*(EY -EK)) /size(EY,1));
MAE4(i)=sum(abs(EY -EK))/size(EY,1);
end

R2,LR2,RR2,PR2
RMSE1,RMSE2,RMSE3,RMSE4
MAE1,MAE2,MAE3,MAE4


Means=[mean(R2),mean(LR2),mean(RR2),mean(PR2);mean(RMSE1),mean(RMSE2),mean(RMSE3),mean(RMSE4);...
    mean(MAE1),mean(MAE2),mean(MAE3),mean(MAE4)]

Std=[std(R2),std(LR2),std(RR2),std(PR2);std(RMSE1),std(RMSE2),std(RMSE3),std(RMSE4);...
    std(MAE1),std(MAE2),std(MAE3),std(MAE4)]



