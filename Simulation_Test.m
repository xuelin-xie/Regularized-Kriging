clc;
clear;
addpath(genpath(pwd));

setdemorandstream(pi);
problem.f=@borehole;  % borehole/steelcol simulation function
[XL,XU]=borehole_bound();  
bounds=[XL;XU];
dim=size(bounds,2); 
pointnum=80;  % 80/100
n_repeats=10;

%% Copyright: Xuelin Xie (xl.xie@whu.edu.cn)

% Predefined variables
[UR2, URMSE1, UMAE1] = deal(zeros(1,n_repeats));
[TRLR2, TRRMSE2, TRMAE2] = deal(zeros(1,n_repeats));
[TRRR2, TRRMSE3, TRMAE3] = deal(zeros(1,n_repeats));
[TRER2, TRRMSE4, TRMAE4] = deal(zeros(1,n_repeats));
[PBLR2, PBRMSE2, PBMAE2] = deal(zeros(1,n_repeats));
[PBRR2, PBRMSE3, PBMAE3] = deal(zeros(1,n_repeats));
[PBER2, PBRMSE4, PBMAE4] = deal(zeros(1,n_repeats));
[t1,t2,t3,t4,t5,t6,t7] = deal(zeros(1,n_repeats));

for i=1:n_repeats

%% Sample point and evaluation point
Samp=LHD(XL,XU,pointnum);  
RealY=callobj(problem.f,Samp);
Data=[Samp,RealY];
n=randsample(pointnum,pointnum*3/4,'false'); 
A=Data(n,:); % A training set
c=1:pointnum;
c(n)=[];
B=Data(c,:); % B testing set
S=A(:,1:end-1);
Y=A(:,end);
EX=B(:,1:end-1);
EY=B(:,end);

%% Finding the optimal regularization parameters
choose = 1;  % (1: Manual adjustment; 2: GSCV Method)
if choose==1
    %%%% !!!! Notice: Borehole function paramaters. %%%% 
    % TR-LK, TR-RK, TR-EK
    TR_bestlambda=0.0032;
    TR_bestmu=0.00316;
    TR_bestalpha=0.9;
    TR_bestgamma=0.0001;
    % PB-LK, PB-RK, PB-EK
    PB_bestlambda=0.00003;
    PB_bestmu=0.1;
    PB_bestalpha=0.25;
    PB_bestgamma=0.01;
else
    TR_bestlambda=OptRPL(S,Y);
    TR_bestmu=OptRPR(S,Y);
    [~,TR_bestalpha,TR_bestgamma] = EPTKGridSearch(S,Y);
    PB_bestlambda=OpbRPL(S,Y);
    PB_bestmu=OpbRPR(S,Y);
    [~,PB_bestalpha,PB_bestgamma] = EPBKGridSearch(S,Y);
end

%% UK
tic; krig1=buildKRG(S,Y); t1(i)=toc;
UK= predictor(EX, krig1);
UR2(i)=1-sum((EY-UK).^2)/sum((EY-mean(EY)).^2);
URMSE1(i)=sqrt(mean((EY-UK).^2));
UMAE1(i)=mean(abs(EY-UK));

%% TR-LK
tic; krig2=buildKRGLPeT(S,Y,TR_bestlambda); t2(i)=toc;
TRLK= predictor(EX, krig2);
TRLR2(i)=1-sum((EY-TRLK).^2)/sum((EY-mean(EY)).^2);
TRRMSE2(i)=sqrt(mean((EY-TRLK).^2));
TRMAE2(i)=mean(abs(EY-TRLK));

%% TR-RK
tic; krig3=buildKRGRPeT(S,Y,TR_bestmu); t3(i)=toc;
TRRK= predictor(EX, krig3);
TRRR2(i)=1-sum((EY-TRRK).^2)/sum((EY-mean(EY)).^2);
TRRMSE3(i)=sqrt(mean((EY-TRRK).^2));
TRMAE3(i)=mean(abs(EY-TRRK));

%% TR-EK
tic; krig4=buildKRGEPeT(S,Y,TR_bestalpha,TR_bestgamma); t4(i)=toc;
EK= predictor(EX, krig4);
TRER2(i)=1-sum((EY-EK).^2)/sum((EY-mean(EY)).^2);
TRRMSE4(i)=sqrt(mean((EY-EK).^2));
TRMAE4(i)=mean(abs(EY-EK));

%% PB-LK
tic; krig5=buildKRGLPeB(S,Y,PB_bestlambda); t5(i)=toc;
PBLK= predictor(EX, krig5);
PBLR2(i)=1-sum((EY-PBLK).^2)/sum((EY-mean(EY)).^2);
PBRMSE2(i)=sqrt(mean((EY-PBLK).^2));
PBMAE2(i)=mean(abs(EY-PBLK));

%% PB-RK
tic; krig6=buildKRGRPeB(S,Y,PB_bestmu); t6(i)=toc;
PBRK= predictor(EX, krig6);
PBRR2(i)=1-sum((EY-PBRK).^2)/sum((EY-mean(EY)).^2);
PBRMSE3(i)=sqrt(mean((EY-PBRK).^2));
PBMAE3(i)=mean(abs(EY-PBRK));

%% PB-EK
tic; krig7=buildKRGEPeB(S,Y,PB_bestalpha,PB_bestgamma); t7(i)=toc;
PBEK= predictor(EX, krig7);
PBER2(i)=1-sum((EY-PBEK).^2)/sum((EY-mean(EY)).^2);
PBRMSE4(i)=sqrt(mean((EY-PBEK).^2));
PBMAE4(i)=mean(abs(EY-PBEK));

end

%% ==================== Results Display ====================
% Method Name
method_names = {'UK', 'TR-LK', 'TR-RK', 'TR-EK', 'PB-LK', 'PB-RK', 'PB-EK'};

% Method Results
R2_results = [UR2; TRLR2; TRRR2; TRER2; PBLR2; PBRR2; PBER2];
RMSE_results = [URMSE1; TRRMSE2; TRRMSE3; TRRMSE4; PBRMSE2; PBRMSE3; PBRMSE4];
MAE_results = [UMAE1; TRMAE2; TRMAE3; TRMAE4; PBMAE2; PBMAE3; PBMAE4];
time_results = [t1; t2; t3; t4; t5; t6; t7];

% Mean and Standard Deviation
R2_mean = mean(R2_results,2);
R2_std = std(R2_results,0,2);
RMSE_mean = mean(RMSE_results,2);
RMSE_std = std(RMSE_results,0,2);
MAE_mean = mean(MAE_results,2);
MAE_std = std(MAE_results,0,2);
time_mean = mean(time_results,2);
time_std = std(time_results,0,2);

%% Results Table
fprintf('\n');
fprintf('================================================================================\n');
fprintf('                         Kriging Model Comparison Results\n');
fprintf('================================================================================\n');
fprintf('Test Function: borehole  |  Sample Size: %d  |  Repeats: %d\n', pointnum, n_repeats);
fprintf('================================================================================\n\n');

% R² Results
fprintf('【R² (Coefficient of Determination) - Higher is Better】\n');
fprintf('--------------------------------------------------------------------\n');
fprintf('%-10s | %12s | %12s | %12s\n', 'Method', 'Mean', 'Std', 'Mean±Std');
fprintf('--------------------------------------------------------------------\n');
for i = 1:7
    fprintf('%-10s | %12.6f | %12.6f | %.6f ± %.6f\n', ...
        method_names{i}, R2_mean(i), R2_std(i), R2_mean(i), R2_std(i));
end
fprintf('--------------------------------------------------------------------\n\n');

% RMSE Results
fprintf('【RMSE (Root Mean Square Error) - Lower is Better】\n');
fprintf('--------------------------------------------------------------------\n');
fprintf('%-10s | %12s | %12s | %12s\n', 'Method', 'Mean', 'Std', 'Mean±Std');
fprintf('--------------------------------------------------------------------\n');
for i = 1:7
    fprintf('%-10s | %12.6e | %12.6e | %.6e ± %.6e\n', ...
        method_names{i}, RMSE_mean(i), RMSE_std(i), RMSE_mean(i), RMSE_std(i));
end
fprintf('--------------------------------------------------------------------\n\n');

% MAE Results
fprintf('【MAE (Mean Absolute Error) - Lower is Better】\n');
fprintf('--------------------------------------------------------------------\n');
fprintf('%-10s | %12s | %12s | %12s\n', 'Method', 'Mean', 'Std', 'Mean±Std');
fprintf('--------------------------------------------------------------------\n');
for i = 1:7
    fprintf('%-10s | %12.6e | %12.6e | %.6e ± %.6e\n', ...
        method_names{i}, MAE_mean(i), MAE_std(i), MAE_mean(i), MAE_std(i));
end
fprintf('--------------------------------------------------------------------\n\n');

% CPU Time
fprintf('【CPU Time (seconds) - Lower is Better】\n');
fprintf('--------------------------------------------------------------------\n');
fprintf('%-10s | %12s | %12s | %12s\n', 'Method', 'Mean', 'Std', 'Mean±Std');
fprintf('--------------------------------------------------------------------\n');
for i = 1:7
    fprintf('%-10s | %12.6f | %12.6f | %.6f ± %.6f\n', ...
        method_names{i}, time_mean(i), time_std(i), time_mean(i), time_std(i));
end
fprintf('--------------------------------------------------------------------\n');

% Find the optimal method
[~, best_R2_idx] = max(R2_mean);
[~, best_RMSE_idx] = min(RMSE_mean);
[~, best_MAE_idx] = min(MAE_mean);
[~, best_time_idx] = min(time_mean);

fprintf('\n【Best Performance Summary】\n');
fprintf('================================================================================\n');
fprintf('✓ Best R²:    %s (%.6f)\n', method_names{best_R2_idx}, R2_mean(best_R2_idx));
fprintf('✓ Best RMSE:  %s (%.6e)\n', method_names{best_RMSE_idx}, RMSE_mean(best_RMSE_idx));
fprintf('✓ Best MAE:   %s (%.6e)\n', method_names{best_MAE_idx}, MAE_mean(best_MAE_idx));
fprintf('✓ Best Speed: %s (%.6f sec)\n', method_names{best_time_idx}, time_mean(best_time_idx));
fprintf('================================================================================\n');

%% Save Results (Optional)
% xlswrite('borehole_80.xlsx', Means, 1);
% xlswrite('borehole_80.xlsx', Std, 2);
% xlswrite('borehole_80.xlsx', time, 3);
