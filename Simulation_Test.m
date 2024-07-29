clc;
clear;
setdemorandstream(pi);
problem.f=@borehole;  % borehole/steelcol simulation function
[XL,XU]=borehole_bound();  
bounds=[XL;XU];
dim=size(bounds,2); 
pointnum=80;  % 80/100

%% Copyright: Xuelin Xie (xl.xie@whu.edu.cn)

for i=1:10

%% Sample point and evaluation point
Samp=LHD(XL,XU,pointnum);  
RealY=callobj(problem.f,Samp);
Data=[Samp,RealY];
n=randsample(pointnum,pointnum*3/4,'false'); 
A=Data(n,:); % A training set
c=1:pointnum;
c(n)=[];
B=Data(c,:);% B testing set
S=A(:,1:end-1);
Y=A(:,end);
EX=B(:,1:end-1);
EY=B(:,end);

%% Finding the optimal regularization parameters
choose = 2;  % (1: Manual  adjustment; 2: GSCV Method) 
%%%%%%  Notice: You can choose one or the other of the two methods %%%%%%%%
if choose==1
    %%% Method 1: Manual  adjustment (Extremely fast, but every time it needs to be manually adjusted)  
    % TR-LK
    TR_bestlambda=0.0032;
    % TR-RK
    TR_bestmu=0.00316;
    % TR-EK
    TR_bestalpha=0.9;
    TR_bestgamma=0.0001;
    % PB-LK
    PB_bestlambda=0.00003;
    % PB-RK
    PB_bestmu=0.1;
    % PB-EK
    PB_bestalpha=0.25;
    PB_bestgamma=0.01;
else
    %%% Method 2: GSCV (Slow, not recommended for high dimensions)
    % TR-LK
    TR_bestlambda=OptRPL(S,Y);
    % TR-RK
    TR_bestmu=OptRPR(S,Y);
    % TR-EK
    [CVmse,TR_bestalpha,TR_bestgamma] = EPTKGridSearch(S,Y);
    % PB-LK
    PB_bestlambda=OpbRPL(S,Y);
    % PB-RK
    PB_bestmu=OpbRPR(S,Y);
    % PB-EK
    [CVmse,PB_bestalpha,PB_bestgamma] = EPBKGridSearch(S,Y);
end

%% UK
tic
krig1=buildKRG(S,Y);
toc1=toc;
t1(i)=sum(toc1);
% predicted values
UK= predictor(EX, krig1);
% The evaluation index of the Kriging model 
UR2(i)=1-sum((EY -UK).*(EY-UK)) /sum((EY-mean(EY)).*(EY-mean(EY)));
URMSE1(i)=sqrt(sum((EY -UK).*(EY -UK)) /size(EY,1));
UMAE1(i)=sum(abs(EY -UK))/size(EY,1);


%% TR-LK
% bulid the model
tic
krig2=buildKRGLPeT(S,Y,TR_bestlambda); 
toc2=toc;
t2(i)=sum(toc2);
% predicted values
TRLK= predictor(EX, krig2);
% The evaluation index of the LK model
TRLR2(i)=1-sum((EY -TRLK).*(EY-TRLK)) /sum((EY-mean(EY)).*(EY-mean(EY)));
TRRMSE2(i)=sqrt(sum((EY -TRLK).*(EY -TRLK)) /size(EY,1));
TRMAE2(i)=sum(abs(EY -TRLK))/size(EY,1);

%% TR-RK
% bulid the model
tic
krig3=buildKRGRPeT(S,Y,TR_bestmu);
toc3=toc;
t3(i)=sum(toc3);
% predicted values
TRRK= predictor(EX, krig3);
% The evaluation index of the RK model
TRRR2(i)=1-sum((EY -TRRK).*(EY-TRRK)) /sum((EY-mean(EY)).*(EY-mean(EY)));
TRRMSE3(i)=sqrt(sum((EY -TRRK).*(EY -TRRK)) /size(EY,1));
TRMAE3(i)=sum(abs(EY -TRRK))/size(EY,1);

%% TR-EK
% bulid the model
tic
krig4=buildKRGEPeT(S,Y,TR_bestalpha,TR_bestgamma);
toc4=toc;
t4(i)=sum(toc4);
% predicted values
EK= predictor(EX, krig4);
% The evaluation index of the EK model
TRER2(i)=1-sum((EY -EK).*(EY-EK)) /sum((EY-mean(EY)).*(EY-mean(EY)));
TRRMSE4(i)=sqrt(sum((EY -EK).*(EY -EK)) /size(EY,1));
TRMAE4(i)=sum(abs(EY -EK))/size(EY,1);

%% PB-LK
% bulid the model
tic
krig5=buildKRGLPeB(S,Y,PB_bestlambda); 
toc5=toc;
t5(i)=sum(toc5);
% predicted values
PBLK= predictor(EX, krig5);
% The evaluation index of the LK model
PBLR2(i)=1-sum((EY -PBLK).*(EY-PBLK)) /sum((EY-mean(EY)).*(EY-mean(EY)));
PBRMSE2(i)=sqrt(sum((EY -PBLK).*(EY -PBLK)) /size(EY,1));
PBMAE2(i)=sum(abs(EY -PBLK))/size(EY,1);

%% PB-RK
% bulid the model
tic
krig6=buildKRGRPeB(S,Y,PB_bestmu);
toc6=toc;
t6(i)=sum(toc6);
% predicted values
PBRK= predictor(EX, krig6);
% The evaluation index of the RK model
PBRR2(i)=1-sum((EY -PBRK).*(EY-PBRK)) /sum((EY-mean(EY)).*(EY-mean(EY)));
PBRMSE3(i)=sqrt(sum((EY -PBRK).*(EY -PBRK)) /size(EY,1));
PBMAE3(i)=sum(abs(EY -PBRK))/size(EY,1);

%% PB-EK
% build the model
tic
krig7=buildKRGEPeB(S,Y,PB_bestalpha,PB_bestgamma);
toc7=toc;
t7(i)=sum(toc7);
% predicted values
PBEK= predictor(EX, krig7);
% The evaluation index of the EK model
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

%% CPU time
time=[mean(t1),mean(t2),mean(t3),mean(t4),mean(t5),mean(t6),mean(t7)]
