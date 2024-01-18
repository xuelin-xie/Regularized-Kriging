clc;
clear;
setdemorandstream(pi);
problem.f=@copeak; % Test functions // copeak/ langermann/ Levy/ morcaf95a/ Sphere/ rothyp/ Tridd / Schwefel....
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

%% UK
krig1=buildKRG(S,Y);  
%% predicted values
UK= predictor(EX, krig1);
%% The evaluation index of the Kriging model 
UR2(i)=1-sum((EY -UK).*(EY-UK)) /sum((EY-mean(EY)).*(EY-mean(EY)));
URMSE1(i)=sqrt(sum((EY -UK).*(EY -UK)) /size(EY,1));
UMAE1(i)=sum(abs(EY -UK))/size(EY,1);

%% TR-LK
%% Obtain optimal parameters
TR_bestlambda=OptRPL(S,Y);
krig2=buildKRGLPeT(S,Y,TR_bestlambda); 
%% predicted values
TRLK= predictor(EX, krig2);
%% The evaluation index of the LK model
TRLR2(i)=1-sum((EY -TRLK).*(EY-TRLK)) /sum((EY-mean(EY)).*(EY-mean(EY)));
TRRMSE2(i)=sqrt(sum((EY -TRLK).*(EY -TRLK)) /size(EY,1));
TRMAE2(i)=sum(abs(EY -TRLK))/size(EY,1);

%% TR-RK
%% Obtain optimal parameters
PT_bestmu=OptRPR(S,Y);
krig3=buildKRGRPeT(S,Y,PT_bestmu);   
%% predicted values
TRRK= predictor(EX, krig3);
%% The evaluation index of the RK model
TRRR2(i)=1-sum((EY -TRRK).*(EY-TRRK)) /sum((EY-mean(EY)).*(EY-mean(EY)));
TRRMSE3(i)=sqrt(sum((EY -TRRK).*(EY -TRRK)) /size(EY,1));
TRMAE3(i)=sum(abs(EY -TRRK))/size(EY,1);

%% TR-EK
% Obtain optimal parameters
[CVmse,PT_bestalpha,PT_bestgamma] = EPTKGridSearch(S,Y,5);
krig4=buildKRGEPeT(S,Y,PT_bestalpha,PT_bestgamma);
%% predicted values
EK= predictor(EX, krig4);
%% The evaluation index of the EK model
TRER2(i)=1-sum((EY -EK).*(EY-EK)) /sum((EY-mean(EY)).*(EY-mean(EY)));
TRRMSE4(i)=sqrt(sum((EY -EK).*(EY -EK)) /size(EY,1));
TRMAE4(i)=sum(abs(EY -EK))/size(EY,1);

%% PB-LK
%% Obtain optimal parameters
PB_bestlambda=OpbRPL(S,Y);
krig5=buildKRGLPeB(S,Y,PB_bestlambda); 
%% predicted values
PBLK= predictor(EX, krig5);
%% The evaluation index of the LK model
PBLR2(i)=1-sum((EY -PBLK).*(EY-PBLK)) /sum((EY-mean(EY)).*(EY-mean(EY)));
PBRMSE2(i)=sqrt(sum((EY -PBLK).*(EY -PBLK)) /size(EY,1));
PBMAE2(i)=sum(abs(EY -PBLK))/size(EY,1);

%% PB-RK
%% Obtain optimal parameters
PB_bestmu=OpbRPR(S,Y);
krig6=buildKRGRPeB(S,Y,PB_bestmu);   
%% predicted values
PBRK= predictor(EX, krig6);
%% The evaluation index of the RK model
PBRR2(i)=1-sum((EY -PBRK).*(EY-PBRK)) /sum((EY-mean(EY)).*(EY-mean(EY)));
PBRMSE3(i)=sqrt(sum((EY -PBRK).*(EY -PBRK)) /size(EY,1));
PBMAE3(i)=sum(abs(EY -PBRK))/size(EY,1);

%% PB-EK
% Obtain optimal parameters
[CVmse,PB_bestalpha,PB_bestgamma] = EPBKGridSearch(S,Y,5);
krig7=buildKRGEPeB(S,Y,PB_bestalpha,PB_bestgamma);
%% predicted values
PBEK= predictor(EX, krig7);
%% The evaluation index of the EK model
PBER2(i)=1-sum((EY -PBEK).*(EY-PBEK)) /sum((EY-mean(EY)).*(EY-mean(EY)));
PBRMSE4(i)=sqrt(sum((EY -PBEK).*(EY -PBEK)) /size(EY,1));
PBMAE4(i)=sum(abs(EY -PBEK))/size(EY,1);
end

UR2,TRLR2,TRRR2,TRER2,PBLR2,PBRR2,PBER2
URMSE1,TRRMSE2,TRRMSE3,TRRMSE4,PBRMSE2,PBRMSE3,PBRMSE4
UMAE1,TRMAE2,TRMAE3,TRMAE4,PBMAE2,PBMAE3,PBMAE4

Means=[mean(UR2),mean(TRLR2),mean(TRRR2),mean(TRER2),mean(PBLR2),mean(PBRR2),mean(PBER2);
    mean(URMSE1),mean(TRRMSE2),mean(TRRMSE3),mean(TRRMSE4),mean(PBRMSE2),mean(PBRMSE3),mean(PBRMSE4);...
    mean(UMAE1),mean(TRMAE2),mean(TRMAE3),mean(TRMAE4),mean(PBMAE2),mean(PBMAE3),mean(PBMAE4)]

Std=[std(UR2),std(TRLR2),std(TRRR2),std(TRER2),std(PBLR2),std(PBRR2),std(PBER2);
    std(URMSE1),std(TRRMSE2),std(TRRMSE3),std(TRRMSE4),std(PBRMSE2),std(PBRMSE3),std(PBRMSE4);...
    std(UMAE1),std(TRMAE2),std(TRMAE3),std(TRMAE4),std(PBMAE2),std(PBMAE3),std(PBMAE4)]

% xlswrite('copeak_60.xlsx',Means,1);
% xlswrite('copeak_60.xlsx',Std,2);

