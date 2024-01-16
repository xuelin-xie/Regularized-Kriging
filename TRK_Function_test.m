clc;
clear;
setdemorandstream(pi);
problem.f=@copeak; % Test functions // copeak/ Drop/ langermann/ morcaf95a/ Sphere/ rothyp/ Tridd / Schwefel....
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

%% Universal Kriging
krig1=buildKRG(S,Y);  
%% predicted values
K= predictor(EX, krig1);
%% The evaluation index of the Kriging model 
UR2(i)=1-sum((EY -K).*(EY-K)) /sum((EY-mean(EY)).*(EY-mean(EY)));
URMSE1(i)=sqrt(sum((EY -K).*(EY -K)) /size(EY,1));
UMAE1(i)=sum(abs(EY -K))/size(EY,1);

%% TR-LK
%% Obtain optimal parameters
bestlambda=OptRPL(S,Y);
krig2=buildKRGLPeT(S,Y,bestlambda); 
%% predicted values
LK= predictor(EX, krig2);
%% The evaluation index of the LK model
TR_LR2(i)=1-sum((EY -LK).*(EY-LK)) /sum((EY-mean(EY)).*(EY-mean(EY)));
TR_RMSE2(i)=sqrt(sum((EY -LK).*(EY -LK)) /size(EY,1));
TR_MAE2(i)=sum(abs(EY -LK))/size(EY,1);


%% TR-RK
%% Obtain optimal parameters
bestmu=OptRPR(S,Y);
krig3=buildKRGRPeT(S,Y,bestmu);   
%% predicted values
RK= predictor(EX, krig3);
%% The evaluation index of the RK model
TR_RR2(i)=1-sum((EY -RK).*(EY-RK)) /sum((EY-mean(EY)).*(EY-mean(EY)));
TR_RMSE3(i)=sqrt(sum((EY -RK).*(EY -RK)) /size(EY,1));
TR_MAE3(i)=sum(abs(EY -RK))/size(EY,1);

%% TR-EK
% Obtain optimal parameters
[CVmse,bestalpha,bestgamma] = EPTKGridSearch(S,Y,5);
krig4=buildKRGEPeT(S,Y,bestalpha,bestgamma);
%% predicted values
EK= predictor(EX, krig4);
%% The evaluation index of the EK model
TR_PR2(i)=1-sum((EY -EK).*(EY-EK)) /sum((EY-mean(EY)).*(EY-mean(EY)));
TR_RMSE4(i)=sqrt(sum((EY -EK).*(EY -EK)) /size(EY,1));
TR_MAE4(i)=sum(abs(EY -EK))/size(EY,1);
end

UR2,TR_LR2,TR_RR2,TR_PR2
URMSE1,TR_RMSE2,TR_RMSE3,TR_RMSE4
UMAE1,TR_MAE2,TR_MAE3,TR_MAE4


Means=[mean(UR2),mean(TR_LR2),mean(TR_RR2),mean(TR_PR2);mean(URMSE1),mean(TR_RMSE2),mean(TR_RMSE3),mean(TR_RMSE4);...
    mean(UMAE1),mean(TR_MAE2),mean(TR_MAE3),mean(TR_MAE4)]

Std=[std(UR2),std(TR_LR2),std(TR_RR2),std(TR_PR2);std(URMSE1),std(TR_RMSE2),std(TR_RMSE3),std(TR_RMSE4);...
    std(UMAE1),std(TR_MAE2),std(TR_MAE3),std(TR_MAE4)]



