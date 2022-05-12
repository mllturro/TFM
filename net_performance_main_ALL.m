%% Utilizing trained DNNs to improve ACB performance

clear all; close all;

% For different cases to be simulated:
%        * Case (1): P_ACB computed using the c1 (N_1,N_2) values
%        * Case (2): P_ACB computed using the DNN-estimated (N_1,N_2)
%        trained on a T_SIB2 = 1 RAO accuracy
%        * Case (3): P_ACB computed using the DNN-estimated (N_1,N_2)
%        values trained on a T_SIB2 = 16 RAOs accuracy
%        * Case (4): P_ACB obtained from a DQL agent trained on a T_SIB2 =
%        1 RAO accuracy
%        * Case (5): Uniform policy

% P_ACB always updated on a T_SIB2 = 16 RAOs time frame as specified by
% 3GPP standards

%% Constants related to 5G environment
env5GConst.T = 10; % Episode or Frame duration in sec.
env5GConst.T_RAO=0.005;    % RAO time in sec.
env5GConst.N_RAO = env5GConst.T/env5GConst.T_RAO;      % Number of RAO channels in a frame
env5GConst.MaxRAO = ceil(2*env5GConst.N_RAO);  % Number of c1 RAO channels during which the system's evolution is observed
env5GConst.N_SIB2 = 16; % no of RAO channels during which state stats (including P_ACB) are not updated

env5GConst.N_steps = round(env5GConst.MaxRAO/env5GConst.N_SIB2); 
% no of times that the P_ACB is broadcast

% *** REMEMBER TO CHANGE ACCORDINGLY ***
env5GConst.MaxWait = 10; % max no of device collisions

env5GConst.Ac_Pacb = 0.05:0.05:1; % Access Class Barring Factor
env5GConst.M = 54;        % Number of available simultaneous communications
env5GConst.N_Dev = 30000;         % Number of MTC devices in scheduled traffic model
env5GConst.Norm = 0;            % Normalize Input, recommended with sigmoide

%% Loading trained net file

% Add file path to where trained nets are stored:
% ** REMOVE WHEN SENDING TO SERVER **
%addpath('C:\Users\mlltu\OneDrive\Documentos\MATT\TFM\Codi Martí\New requests NN estimation\Trained nets');

% Net file for Case (2):
FileName_net = strcat('trained_net_21_1_1_1_Max_10');

load(FileName_net,'net');
load(FileName_net,'mu');
load(FileName_net,'sigma');

% save normalization parameters as object 'net_param'
net_c2 = net;
net_param_c2.mu = mu;
net_param_c2.sigma = sigma;

% Net file for Case (3):
FileName_net = strcat('trained_net_21_16_1_2_Max_10');

load(FileName_net,'net');
load(FileName_net,'mu');
load(FileName_net,'sigma');

% save normalization parameters as object 'net_param'
net_c3 = net;
net_param_c3.mu = mu;
net_param_c3.sigma = sigma;

%% Loading trained DQL agent

% Add file path to where trained DQL agents are stored:
% ** REMOVE WHEN SENDING TO SERVER **
%addpath('C:\Users\mlltu\OneDrive\Documentos\MATT\TFM\Memòria\DQL agent training evolution\');

% REMINDER: till the moment, only DQL agents trained with MaxWait = 10 are
% available
agent_ID = 7501; % highest absolute reward
agent_ID_2 = 7504; % highest avg. reward 
FileName_DQL_1 = strcat('Agent',num2str(agent_ID));
FileName_DQL_2 = strcat('Agent',num2str(agent_ID_2));

saved_agent_1 = load(FileName_DQL_1,'saved_agent'); 
saved_agent_1 = saved_agent_1.saved_agent;

saved_agent_2 = load(FileName_DQL_2,'saved_agent'); 
saved_agent_2 = saved_agent_2.saved_agent;

%% Episode simulation
% No. of simulations that are averaged
no_it = 1e3;

St_Mat.c1 = zeros(no_it,env5GConst.MaxRAO,8);
St_Mat.c2 = zeros(no_it,env5GConst.MaxRAO,8); 
St_Mat.c3 = zeros(no_it,env5GConst.MaxRAO,8);
St_Mat.c4 = zeros(no_it,env5GConst.MaxRAO,8);
St_Mat.c5 = zeros(no_it,env5GConst.MaxRAO,8);
St_Mat.c6 = zeros(no_it,env5GConst.MaxRAO,8);

Delays.c1 = []; Delays.c2 = []; Delays.c3 = []; Delays.c4 = []; Delays.c5 = []; Delays.c6 = [];
Avg_Delay.c1 = 0; Avg_Delay.c2 = 0; Avg_Delay.c3 = 0; Avg_Delay.c4 = 0; Avg_Delay.c5 = 0; Avg_Delay.c6 = 0; 
for it = 1:no_it
    [MTC_feat_it,St_Mat_it] = epi_sim_estimated_action(env5GConst,net_c2,net_c3,net_param_c2,net_param_c3,saved_agent_1,saved_agent_2);
    St_Mat.c1(it,:,:) = St_Mat_it.c1; 
    St_Mat.c2(it,:,:) = St_Mat_it.c2;
    St_Mat.c3(it,:,:) = St_Mat_it.c3;
    St_Mat.c4(it,:,:) = St_Mat_it.c4;
    St_Mat.c5(it,:,:) = St_Mat_it.c5;
    St_Mat.c6(it,:,:) = St_Mat_it.c6;

    Delays.c1 = [Delays.c1 MTC_feat_it.c1(MTC_feat_it.c1(:,5) == 1,2)']; 
    Avg_Delay.c1 = Avg_Delay.c1 + mean(MTC_feat_it.c1(MTC_feat_it.c1(:,5) == 1,2));

    Delays.c2 = [Delays.c2 MTC_feat_it.c2(MTC_feat_it.c2(:,5) == 1,2)']; 
    Avg_Delay.c2 = Avg_Delay.c2 + mean(MTC_feat_it.c2(MTC_feat_it.c2(:,5) == 1,2));

    Delays.c3 = [Delays.c3 MTC_feat_it.c3(MTC_feat_it.c3(:,5) == 1,2)']; 
    Avg_Delay.c3 = Avg_Delay.c3 + mean(MTC_feat_it.c3(MTC_feat_it.c3(:,5) == 1,2));

    Delays.c4 = [Delays.c4 MTC_feat_it.c4(MTC_feat_it.c4(:,5) == 1,2)']; 
    Avg_Delay.c4 = Avg_Delay.c4 + mean(MTC_feat_it.c4(MTC_feat_it.c4(:,5) == 1,2));

    Delays.c5 = [Delays.c5 MTC_feat_it.c5(MTC_feat_it.c5(:,5) == 1,2)']; 
    Avg_Delay.c5 = Avg_Delay.c5 + mean(MTC_feat_it.c5(MTC_feat_it.c5(:,5) == 1,2));

    Delays.c6 = [Delays.c6 MTC_feat_it.c6(MTC_feat_it.c6(:,5) == 1,2)']; 
    Avg_Delay.c6 = Avg_Delay.c6 + mean(MTC_feat_it.c6(MTC_feat_it.c5(:,5) == 1,2));

    disp(strcat('Episode Loop - Iteration No. : ',num2str(it)));
end
Avg_Delay.c1 = Avg_Delay.c1/no_it;
Avg_Delay.c2 = Avg_Delay.c2/no_it;
Avg_Delay.c3 = Avg_Delay.c3/no_it;
Avg_Delay.c4 = Avg_Delay.c4/no_it;
Avg_Delay.c5 = Avg_Delay.c5/no_it;
Avg_Delay.c6 = Avg_Delay.c6/no_it;
Avg_Delay



%% Avergage state stats over all episodes
St_Mat_avg.c1 = zeros(env5GConst.MaxRAO,8); 
St_Mat_avg.c2 = zeros(env5GConst.MaxRAO,8); 
St_Mat_avg.c3 = zeros(env5GConst.MaxRAO,8);
St_Mat_avg.c4 = zeros(env5GConst.MaxRAO,8);
St_Mat_avg.c5 = zeros(env5GConst.MaxRAO,8);
St_Mat_avg.c6 = zeros(env5GConst.MaxRAO,8);

Abs_Error_c2 = zeros(8,2,env5GConst.MaxRAO);
Abs_Error_c3 = zeros(8,2,env5GConst.MaxRAO);
Abs_Error_c4 = zeros(8,2,env5GConst.MaxRAO);
Abs_Error_c5 = zeros(8,2,env5GConst.MaxRAO);

% Dimensions = [no. of features; max + min + avg; total no. of RAO
% channels]

for ii = 1:env5GConst.MaxRAO
    % focus on single RAO channel, computing its average over the different 
    % iterations:
    St_Mat_avg.c1(ii,:) = mean(St_Mat.c1(:,ii,:),1); 
    % St_Mat.c1 Dim. = [no_it x no. RAO channels x no. of features]
    St_Mat_avg.c2(ii,:) = mean(St_Mat.c2(:,ii,:),1);
    St_Mat_avg.c3(ii,:) = mean(St_Mat.c3(:,ii,:),1);
    St_Mat_avg.c4(ii,:) = mean(St_Mat.c4(:,ii,:),1);
    St_Mat_avg.c5(ii,:) = mean(St_Mat.c5(:,ii,:),1);
    St_Mat_avg.c6(ii,:) = mean(St_Mat.c6(:,ii,:),1);


    % MAE between the net- or DQL-obtained state stats
    err_ii_c2 = abs(St_Mat.c1(:,ii,:) - St_Mat.c2(:,ii,:));
    err_ii_c3 = abs(St_Mat.c1(:,ii,:) - St_Mat.c3(:,ii,:));
    err_ii_c4 = abs(St_Mat.c1(:,ii,:) - St_Mat.c4(:,ii,:));
    err_ii_c5 = abs(St_Mat.c1(:,ii,:) - St_Mat.c5(:,ii,:));

    % record the average error
    Abs_Error_c2(:,1,ii) = mean(err_ii_c2,1);
    Abs_Error_c3(:,1,ii) = mean(err_ii_c3,1);
    Abs_Error_c4(:,1,ii) = mean(err_ii_c4,1);
    Abs_Error_c5(:,1,ii) = mean(err_ii_c5,1);

    % record the maximum error
    Abs_Error_c2(:,2,ii) = max(err_ii_c2,[],1);
    Abs_Error_c3(:,2,ii) = max(err_ii_c3,[],1);
    Abs_Error_c4(:,2,ii) = max(err_ii_c4,[],1);
    Abs_Error_c5(:,2,ii) = max(err_ii_c5,[],1);
end

%% Obtain the delay pdf & cdf from episode iterations:

% cdf:
Del_cdf.c1=histogram(Delays.c1,'numbins',100,'Normalization','cdf');
Del_cdf.c1=[Del_cdf.c1.BinEdges(2:Del_cdf.c1.NumBins+1);Del_cdf.c1.Values];

Del_cdf.c2=histogram(Delays.c2,'numbins',100,'Normalization','cdf');
Del_cdf.c2=[Del_cdf.c2.BinEdges(2:Del_cdf.c2.NumBins+1);Del_cdf.c2.Values];

Del_cdf.c3=histogram(Delays.c3,'numbins',100,'Normalization','cdf');
Del_cdf.c3=[Del_cdf.c3.BinEdges(2:Del_cdf.c3.NumBins+1);Del_cdf.c3.Values];

Del_cdf.c4=histogram(Delays.c4,'numbins',100,'Normalization','cdf');
Del_cdf.c4=[Del_cdf.c4.BinEdges(2:Del_cdf.c4.NumBins+1);Del_cdf.c4.Values];

Del_cdf.c5=histogram(Delays.c5,'numbins',100,'Normalization','cdf');
Del_cdf.c5=[Del_cdf.c5.BinEdges(2:Del_cdf.c5.NumBins+1);Del_cdf.c5.Values];

Del_cdf.c6=histogram(Delays.c6,'numbins',100,'Normalization','cdf');
Del_cdf.c6=[Del_cdf.c6.BinEdges(2:Del_cdf.c6.NumBins+1);Del_cdf.c6.Values];

% pdf:
Del_pdf.c1=histogram(Delays.c1,'numbins',100);
Del_pdf.c1=[Del_pdf.c1.BinEdges(2:Del_pdf.c1.NumBins+1);Del_pdf.c1.Values/sum(Del_pdf.c1.Values)];

Del_pdf.c2=histogram(Delays.c2,'numbins',100);
Del_pdf.c2=[Del_pdf.c2.BinEdges(2:Del_pdf.c2.NumBins+1);Del_pdf.c2.Values/sum(Del_pdf.c2.Values)];

Del_pdf.c3=histogram(Delays.c3,'numbins',100);
Del_pdf.c3=[Del_pdf.c3.BinEdges(2:Del_pdf.c3.NumBins+1);Del_pdf.c3.Values/sum(Del_pdf.c3.Values)];

Del_pdf.c4=histogram(Delays.c4,'numbins',100);
Del_pdf.c4=[Del_pdf.c4.BinEdges(2:Del_pdf.c4.NumBins+1);Del_pdf.c4.Values/sum(Del_pdf.c4.Values)];

Del_pdf.c5=histogram(Delays.c5,'numbins',100);
Del_pdf.c5=[Del_pdf.c5.BinEdges(2:Del_pdf.c5.NumBins+1);Del_pdf.c5.Values/sum(Del_pdf.c5.Values)];

Del_pdf.c6=histogram(Delays.c6,'numbins',100);
Del_pdf.c6=[Del_pdf.c6.BinEdges(2:Del_pdf.c6.NumBins+1);Del_pdf.c6.Values/sum(Del_pdf.c6.Values)];

% Success rate:
succ_rate.c1 = (sum(St_Mat_avg.c1(:,1)) + sum(St_Mat_avg.c1(:,2)))/env5GConst.N_Dev;
succ_rate.c2 = (sum(St_Mat_avg.c2(:,1)) + sum(St_Mat_avg.c2(:,2)))/env5GConst.N_Dev;
succ_rate.c3 = (sum(St_Mat_avg.c3(:,1)) + sum(St_Mat_avg.c3(:,2)))/env5GConst.N_Dev;
succ_rate.c4 = (sum(St_Mat_avg.c4(:,1)) + sum(St_Mat_avg.c4(:,2)))/env5GConst.N_Dev;
succ_rate.c5 = (sum(St_Mat_avg.c5(:,1)) + sum(St_Mat_avg.c5(:,2)))/env5GConst.N_Dev;
succ_rate.c6 = (sum(St_Mat_avg.c6(:,1)) + sum(St_Mat_avg.c6(:,2)))/env5GConst.N_Dev;
succ_rate

% Max Delay shown by 95% of Devices [RAO Channels]
%Max_Delay = min(Del_cdf(1, Del_cdf(2,:) >= 0.95)) % this to compute the
%maximum delay value shown by 95% of devices; originally computed this way,
%though reconverted into the below expression (just like Marga's)
Max_Delay_95.c1 = max(Del_cdf.c1(1, Del_cdf.c1(2,:) < 0.95));
Max_Delay_95.c2 = max(Del_cdf.c2(1, Del_cdf.c2(2,:) < 0.95));
Max_Delay_95.c3 = max(Del_cdf.c3(1, Del_cdf.c3(2,:) < 0.95));
Max_Delay_95.c4 = max(Del_cdf.c4(1, Del_cdf.c4(2,:) < 0.95));
Max_Delay_95.c5 = max(Del_cdf.c5(1, Del_cdf.c5(2,:) < 0.95));
Max_Delay_95.c6 = max(Del_cdf.c6(1, Del_cdf.c6(2,:) < 0.95));
Max_Delay_95

%% Plots
figure(1)

subplot(1,2,1);
% n_1 Error

%{
plot(1:env5GConst.MaxRAO,[St_Mat_avg.c1(:,6)'; St_Mat_avg.c2(:,6)'; St_Mat_avg.c3(:,6)']);
xlabel('RAO Channel'); title('n_1');
legend('c1','c2 by 1 SIB2 Slot','Predicted via NN','location','best'); 
%}

plot(1:env5GConst.MaxRAO,squeeze(Abs_Error_c2(6,1,:)),'DisplayName','T_{SIB2} = 1'); hold on;
plot(1:env5GConst.MaxRAO,squeeze(Abs_Error_c3(6,1,:)),'DisplayName','T_{SIB2} = 16'); hold off;
legend('location','best'); grid on;
title('(a)','Interpreter','latex');
xlabel('RAO','Interpreter','latex'); 
ylabel('Mean Absolute Error (MAE) for $$\hat{n}_1$$','Interpreter','latex');


% n_2 Error

subplot(1,2,2);

plot(1:env5GConst.MaxRAO,squeeze(Abs_Error_c2(7,1,:)),'DisplayName','T_{SIB2} = 1'); hold on;
plot(1:env5GConst.MaxRAO,squeeze(Abs_Error_c3(7,1,:)),'DisplayName','T_{SIB2} = 16'); hold off;
legend('location','best'); grid on;
title('(b)','Interpreter','latex');
xlabel('RAO','Interpreter','latex'); 
ylabel('Mean Absolute Error (MAE) for $$\hat{n}_2$$','Interpreter','latex'); 


%saveas(1,'n_1_n_2_Error_N_max_10.png');

%%
figure(2)

% P_ACB
%subplot(1,2,1);
plot(1:env5GConst.MaxRAO,[St_Mat_avg.c1(:,8)'; St_Mat_avg.c4(:,8)'; St_Mat_avg.c5(:,8)']);
xlabel('RAO','Interpreter','latex'); ylabel('$$p$$','Interpreter','latex'); 
grid on;
title('$$N_u^{max} = 5$$','Interpreter','latex');
legend('Analytical','Agent 7501','Agent 7504','location','best'); 

% P_ACB Error
%{
subplot(1,2,2);
plot(1:env5GConst.MaxRAO,squeeze(Abs_Error_c2(8,1,:)),'DisplayName','Case (2)'); hold on;
plot(1:env5GConst.MaxRAO,squeeze(Abs_Error_c3(8,1,:)),'DisplayName','Case (3)'); hold on;
plot(1:env5GConst.MaxRAO,squeeze(Abs_Error_c4(8,1,:)),'DisplayName','Case (4)'); hold off;
legend('location','best'); grid on;
title('(b)','Interpreter','latex');
xlabel('RAO','Interpreter','latex'); 
ylabel('Mean Absolute Error (MAE) for $$\hat{p}$$','Interpreter','latex');
%}

%saveas(2,'P_ACB.png');

%%
figure(3)

subplot(2,2,1);

% no. of served MTCDs
plot(1:env5GConst.MaxRAO,[St_Mat_avg.c1(:,1)' + St_Mat_avg.c1(:,2)'; ...
    St_Mat_avg.c2(:,1)' + St_Mat_avg.c2(:,2)'; ...
    St_Mat_avg.c3(:,1)' + St_Mat_avg.c3(:,2)';...
    St_Mat_avg.c4(:,1)' + St_Mat_avg.c4(:,2)';...
    St_Mat_avg.c5(:,1)' + St_Mat_avg.c5(:,2)';...
    St_Mat_avg.c6(:,1)' + St_Mat_avg.c6(:,2)']); 
grid on;
legend('Case (1)','Case (2)','Case (3)','Case (4)','Case (5)','Uniform','location','best'); 
title('(a)','Interpreter','latex');
xlabel('RAO','Interpreter','latex'); ylabel('$$N^s$$','Interpreter','latex');


subplot(2,2,2);
% no. of reported collisions

plot(1:env5GConst.MaxRAO,[St_Mat_avg.c1(:,4)'; St_Mat_avg.c2(:,4)';...
    St_Mat_avg.c3(:,4)'; St_Mat_avg.c4(:,4)'; St_Mat_avg.c5(:,4)'; St_Mat_avg.c6(:,4)']); grid on;
legend('Case (1)','Case (2)','Case (3)','Case (4)','Case (5)','Uniform','location','best'); 
title('(b)','Interpreter','latex');
xlabel('RAO','Interpreter','latex'); ylabel('$$N^c$$','Interpreter','latex');

subplot(2,2,3);
% evolution of total accumulated average delay

plot(1:env5GConst.MaxRAO,[St_Mat_avg.c1(:,5)'; St_Mat_avg.c2(:,5)'; ...
    St_Mat_avg.c3(:,5)'; St_Mat_avg.c4(:,5)'; St_Mat_avg.c5(:,5)'; St_Mat_avg.c6(:,5)']); grid on;
legend('Case (1)','Case (2)','Case (3)','Case (4)','Case (5)','Uniform','location','best'); 
title('(c)','Interpreter','latex');
xlabel('RAO','Interpreter','latex'); ylabel('$$D_{avg}$$','Interpreter','latex');

subplot(2,2,4);
% cdf

plot(Del_cdf.c1(1,:),[Del_cdf.c1(2,:); Del_cdf.c2(2,:);...
    Del_cdf.c3(2,:); Del_cdf.c4(2,:); Del_cdf.c5(2,:); Del_cdf.c6(2,:)]); 
grid on;
legend('Case (1)','Case (2)','Case (3)','Case (4)','Case (5)','Uniform','location','best'); 
xlabel('Delay in RAOs','Interpreter','latex'); 
ylabel('Delay CDF','Interpreter','latex')
title('(d)','Interpreter','latex');

%saveas(3,'served_collisions_delay_cdf.png');

%% Saving the file
FileName = strcat('NN_learnt_P_ACB_Cases_Max_10');
save(FileName);

