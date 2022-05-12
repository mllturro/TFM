function [MTC_feat,St_Mat] = epi_sim_estimated_action(env5GConst,net_c2,...
    net_c3,net_param_c2,net_param_c3,DQL_agent_1,DQL_agent_2)
% A trained agent is used to test the trained Q function.
%*************************************************
%    Input:
% env5G        = RL environment object as defined by MATLAB's RL Toolbox
% net          = net object as defined by MATLAB's Deep Learning Toolbbox
%                (contains info on the DNN strcture and associated
%                weights)
% net_param    = contains the parameters needed for normalizing the net's
%                input data
% DQL_agent_1    = DQL agent object (contains a certain DQL-learnt policy)

%*************************************************
%  Output:
% MTC_feat     = N_Dev x 5 matrix containing the system's current stats. 
%                Specifically, the columns are distributed as follows:
%
%                  1) MTC Device number
%                  2) Delay in number of RAO channels
%                  3) number of failed ACB checks (Nbarring)
%                  4) number of collisions
%                  5) 0 / 1(served)
%
% St_Mat        = MaxRAO x 8 matrix containing state space stats (SIB2 
%                 Slot accuracy). The columns are distributed as follows: 
%
%                  1) Ns1 = no. of devices that successfully completed the 
%                           RA procedure that did not previously collide
%                  2) Ns2 = no. of devices that successfully completed the
%                           RA procedure that previously collided
%                  3) No of delayed devices
%                  4) No of collisions
%                  5) Avg Delay
%                  6) n1 = no. of access requests from devies trying to
%                  either pass the ACB check and/or compete for a preamble
%                  that did not previously collide
%                  7) n2 = no of devices competing for a preamble that
%                          previously collided
%                  8) P_ACB (action)
%
%*************************************************

%% Initializing main parameters and containers
Beta_object = makedist('beta','a',3,'b',4);

% create object 'MTC_feat' containing the different state stats using the
% following (n_1, n_2, P_ACB) data:
%
%        * Case (1): P_ACB computed using the c1 (N_1,N_2) values
%        * Case (2): P_ACB computed using the DNN-estimated (N_1,N_2)
%        trained on a T_SIB2 = 1 RAO accuracy
%        * Case (3): P_ACB computed using the DNN-estimated (N_1,N_2)
%        values trained on a T_SIB2 = 16 RAOs accuracy
%        * Case (4): P_ACB obtained from a DQL agent trained on a T_SIB2 =
%        1 RAO accuracy (DQL_agent_1)
%        * Case (5): P_ACB obtained from a DQL agent trained on a T_SIB2 =
%        1 RAO accuracy (DQL_agent_2)
%        * Case (6): uniform policy (P_ACB = 1)

MTC_feat.c1 = zeros(env5GConst.N_Dev,5); % MTC devices are ordered from 1 to Nsch
MTC_feat.c1(:,1) = 1:env5GConst.N_Dev;

MTC_feat.c2 = zeros(env5GConst.N_Dev,5); 
MTC_feat.c2(:,1) = 1:env5GConst.N_Dev;

MTC_feat.c3 = zeros(env5GConst.N_Dev,5); 
MTC_feat.c3(:,1) = 1:env5GConst.N_Dev;

MTC_feat.c4 = zeros(env5GConst.N_Dev,5); 
MTC_feat.c4(:,1) = 1:env5GConst.N_Dev;

MTC_feat.c5 = zeros(env5GConst.N_Dev,5); 
MTC_feat.c5(:,1) = 1:env5GConst.N_Dev;

MTC_feat.c6 = zeros(env5GConst.N_Dev,5); 
MTC_feat.c6(:,1) = 1:env5GConst.N_Dev;

St_Mat.c1 = zeros(env5GConst.MaxRAO,8);
St_Mat.c2 = zeros(env5GConst.MaxRAO,8);
St_Mat.c3 = zeros(env5GConst.MaxRAO,8);
St_Mat.c4 = zeros(env5GConst.MaxRAO,8);
St_Mat.c5 = zeros(env5GConst.MaxRAO,8);
St_Mat.c6 = zeros(env5GConst.MaxRAO,8);

% Initialization objects and parameters needed to use the trained DQL
% agent:
critic_1 = getCritic(DQL_agent_1);
QTable_1=zeros(1,length(env5GConst.Ac_Pacb));
s_1 = [0 0 0 1];

critic_2 = getCritic(DQL_agent_2);
QTable_2=zeros(1,length(env5GConst.Ac_Pacb));
s_2 = [0 0 0 1];

%% Generate access time
% Per-device allocation of a RAO channel no
Beta_vars=random(Beta_object,[1,env5GConst.N_Dev])*env5GConst.T;

MTC_RAOslot.c1 = ceil(Beta_vars*env5GConst.N_RAO/env5GConst.T); 
MTC_RAOslot.c2 = ceil(Beta_vars*env5GConst.N_RAO/env5GConst.T); 
MTC_RAOslot.c3 = ceil(Beta_vars*env5GConst.N_RAO/env5GConst.T);
MTC_RAOslot.c4 = ceil(Beta_vars*env5GConst.N_RAO/env5GConst.T); 
MTC_RAOslot.c5 = ceil(Beta_vars*env5GConst.N_RAO/env5GConst.T); 
MTC_RAOslot.c6 = ceil(Beta_vars*env5GConst.N_RAO/env5GConst.T); 

% Initialize the P_ACB at 1 (the first SIB2 slot will feature this value):
P_ACB.c1 = 1; P_ACB.c2 = 1; P_ACB.c3 = 1; P_ACB.c4 = 1; P_ACB.c5 = 1; P_ACB.c6 = 1;

%% Episode Simulation

for nn = 1:env5GConst.N_steps
    % Initializing the inner loop parameters:
        
    % Resetting the counter before beginning the outer loop (state and action
    % update frame)
    %St_Mat_nn = zeros(1,8); St_Mat_nn_est = zeros(1,8);
        
    % begin inner loop
    for jj = 1:env5GConst.N_SIB2

        % Active MTCDs indices in each RAO:
        Index_request.c1 = MTC_feat.c1(MTC_RAOslot.c1 == (env5GConst.N_SIB2*(nn-1))+jj,1);
        Index_request.c2 = MTC_feat.c2(MTC_RAOslot.c2 == (env5GConst.N_SIB2*(nn-1))+jj,1);
        Index_request.c3 = MTC_feat.c3(MTC_RAOslot.c3 == (env5GConst.N_SIB2*(nn-1))+jj,1);
        Index_request.c4 = MTC_feat.c4(MTC_RAOslot.c4 == (env5GConst.N_SIB2*(nn-1))+jj,1);
        Index_request.c5 = MTC_feat.c5(MTC_RAOslot.c5 == (env5GConst.N_SIB2*(nn-1))+jj,1);
        Index_request.c6 = MTC_feat.c6(MTC_RAOslot.c6 == (env5GConst.N_SIB2*(nn-1))+jj,1);

        [St_Mat_c1_jj, MTC_feat.c1, MTC_RAOslot.c1] = ...
            single_RAO_loop(env5GConst, MTC_feat.c1, ...
            MTC_RAOslot.c1, Index_request.c1, P_ACB.c1);
        [St_Mat_c2_jj, MTC_feat.c2, MTC_RAOslot.c2] = ...
            single_RAO_loop(env5GConst, MTC_feat.c2, ...
            MTC_RAOslot.c2, Index_request.c2, P_ACB.c2);
        [St_Mat_c3_jj, MTC_feat.c3, MTC_RAOslot.c3] = ...
            single_RAO_loop(env5GConst, MTC_feat.c3, ...
            MTC_RAOslot.c3, Index_request.c3, P_ACB.c3);
        [St_Mat_c4_jj, MTC_feat.c4, MTC_RAOslot.c4] = ...
            single_RAO_loop(env5GConst, MTC_feat.c4, ...
            MTC_RAOslot.c4, Index_request.c4, P_ACB.c4);
        [St_Mat_c5_jj, MTC_feat.c5, MTC_RAOslot.c5] = ...
            single_RAO_loop(env5GConst, MTC_feat.c5, ...
            MTC_RAOslot.c5, Index_request.c5, P_ACB.c5);
        [St_Mat_c6_jj, MTC_feat.c6, MTC_RAOslot.c6] = ...
            single_RAO_loop(env5GConst, MTC_feat.c6, ...
            MTC_RAOslot.c6, Index_request.c6, P_ACB.c6);

        St_Mat.c1(env5GConst.N_SIB2*(nn-1)+jj,1:7) = St_Mat_c1_jj;
        St_Mat.c1(env5GConst.N_SIB2*(nn-1)+jj,8) = P_ACB.c1;

        St_Mat.c2(env5GConst.N_SIB2*(nn-1)+jj,1:7) = St_Mat_c2_jj;
        St_Mat.c3(env5GConst.N_SIB2*(nn-1)+jj,1:7) = St_Mat_c3_jj;
        St_Mat.c4(env5GConst.N_SIB2*(nn-1)+jj,1:7) = St_Mat_c4_jj;
        St_Mat.c5(env5GConst.N_SIB2*(nn-1)+jj,1:7) = St_Mat_c5_jj;
        St_Mat.c6(env5GConst.N_SIB2*(nn-1)+jj,1:7) = St_Mat_c6_jj;

        n1_jj = St_Mat_c1_jj(6); n2_jj = St_Mat_c1_jj(7);

        if n1_jj > 0
            P_ACB.c1 = max(0,min(1,(env5GConst.M*(env5GConst.M-1-n2_jj))/...
            (n1_jj*(env5GConst.M-1)-n2_jj)));
        else 
            P_ACB.c1 = 1;
        end

    end

    %St_Mat(nn,1:7) = St_Mat_nn(1:7)./env5GConst.N_SIB2;

    % The P_ACB is kept the same for T_SIB2 consecutive RAO channels
    St_Mat_c2_nn = zeros(1,7); St_Mat_c3_nn = zeros(1,7); St_Mat_c4_nn = zeros(1,7); St_Mat_c5_nn = zeros(1,7);

    for jj = 1:env5GConst.N_SIB2
        % Record the P_ACB used in the past SIB2 RAO channels
        St_Mat.c2(env5GConst.N_SIB2*(nn-1)+jj,8) = P_ACB.c2;
        St_Mat.c3(env5GConst.N_SIB2*(nn-1)+jj,8) = P_ACB.c3;
        St_Mat.c4(env5GConst.N_SIB2*(nn-1)+jj,8) = P_ACB.c4;
        St_Mat.c5(env5GConst.N_SIB2*(nn-1)+jj,8) = P_ACB.c5;

        % Compute the averaged state stats (over a SIB2 period) to plug 
        % them into the trained net or DQL agents, and to update the P_ACB
        St_Mat_c2_nn = ... 
            St_Mat_c2_nn + St_Mat.c2(env5GConst.N_SIB2*(nn-1)+jj,1:7);
        St_Mat_c3_nn = ... 
            St_Mat_c3_nn + St_Mat.c3(env5GConst.N_SIB2*(nn-1)+jj,1:7);
        St_Mat_c4_nn = ... 
            St_Mat_c4_nn + St_Mat.c4(env5GConst.N_SIB2*(nn-1)+jj,1:7);
        St_Mat_c5_nn = ... 
            St_Mat_c5_nn + St_Mat.c5(env5GConst.N_SIB2*(nn-1)+jj,1:7);
    end

    St_Mat_c2_nn = St_Mat_c2_nn./env5GConst.N_SIB2;
    St_Mat_c3_nn = St_Mat_c3_nn./env5GConst.N_SIB2;
    St_Mat_c4_nn = St_Mat_c4_nn./env5GConst.N_SIB2;
    St_Mat_c5_nn = St_Mat_c5_nn./env5GConst.N_SIB2;

    % generate and normalize input data for the net in case (2)
    input_data = [St_Mat_c2_nn(1:2) St_Mat_c2_nn(4:5) P_ACB.c2];
    input_data = (input_data - net_param_c2.mu(1:5))./net_param_c2.sigma(1:5);

    % estimated values of n_1,n_2
    n_nn_c2 = predict(net_c2,input_data);

    % generate and normalize input data for the net in case (2)
    input_data = [St_Mat_c3_nn(1:2) St_Mat_c3_nn(4:5) P_ACB.c3];
    input_data = (input_data - net_param_c3.mu(1:5))./net_param_c3.sigma(1:5);

    % estimated values of n_1,n_2
    n_nn_c3 = predict(net_c3,input_data);

    % de-normalize output values
    n_nn_c2 = n_nn_c2.*net_param_c2.sigma(6:7) + net_param_c2.mu(6:7);
    n_nn_c3 = n_nn_c3.*net_param_c3.sigma(6:7) + net_param_c3.mu(6:7);

    n1_nn_c2 = n_nn_c2(1); n2_nn_c2 = n_nn_c2(2); 
    n1_nn_c3 = n_nn_c3(1); n2_nn_c3 = n_nn_c3(2); 

    for jj = 1:env5GConst.N_SIB2
        St_Mat.c2(env5GConst.N_SIB2*(nn-1)+jj,6:7) = n_nn_c2;
        St_Mat.c3(env5GConst.N_SIB2*(nn-1)+jj,6:7) = n_nn_c3;
    end

    % Case (2) & (3): update the P_ACB using the DNN-obtained (n_1,n_2)
    if n1_nn_c2 > 0
        P_ACB.c2 = max(0,min(1,(env5GConst.M*(env5GConst.M-1-n2_nn_c2))/...
            (n1_nn_c2*(env5GConst.M-1)-n2_nn_c2)));
    else 
        P_ACB.c2 = 1;
    end

    if n1_nn_c3 > 0
        P_ACB.c3 = max(0,min(1,(env5GConst.M*(env5GConst.M-1-n2_nn_c3))/...
            (n1_nn_c3*(env5GConst.M-1)-n2_nn_c3)));
    else 
        P_ACB.c3 = 1;
    end

    % Cases (4) & (5): update the P_ACB using the trained DQL agent
    % ** Note that the DQL agent does not differentitate between types of
    %    successfully connected MTCDs ( N^s = N^s_1 + N^s_2 )
    
    % Case (4):
    for i1=1:length(env5GConst.Ac_Pacb)
        QTable_1(i1) = getValue(critic_1,s_1,env5GConst.Ac_Pacb(i1));
    end
    [~,Acc] = max(QTable_1);
    P_ACB.c4 = env5GConst.Ac_Pacb(Acc);

    s_1 = [St_Mat_c4_nn(1) + St_Mat_c4_nn(2); St_Mat_c4_nn(4);...
        St_Mat_c4_nn(5); P_ACB.c4];

    % Case (5):
    for i1=1:length(env5GConst.Ac_Pacb)
        QTable_2(i1) = getValue(critic_2,s_2,env5GConst.Ac_Pacb(i1));
    end
    [~,Acc] = max(QTable_2);
    P_ACB.c5 = env5GConst.Ac_Pacb(Acc);

    s_2 = [St_Mat_c5_nn(1) + St_Mat_c5_nn(2); St_Mat_c5_nn(4);...
        St_Mat_c5_nn(5); P_ACB.c5];
  
    % Normalize state matrix:
    s_1 = [(s_1(1)*2)/(env5GConst.M); ...
        s_1(2)/(env5GConst.M); ...
        (s_1(3)*2)/(env5GConst.N_steps); ...
        s_1(4)];
    s_2 = [(s_2(1)*2)/(env5GConst.M); ...
        s_2(2)/(env5GConst.M); ...
        (s_2(3)*2)/(env5GConst.N_steps); ...
        s_2(4)];

end