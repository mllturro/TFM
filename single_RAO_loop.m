function [RAO_state, MTC_feat, MTC_RAOslot] = single_RAO_loop(env5GConst, MTC_feat, MTC_RAOslot, Index_request, P_ACB)
    % INPUT
    % * (Index_request)
    % * MTC_feat
    % * MTC_RAO_slot
    % * Index_request
    %  OUTPUT
    % * MTC_feat
    % * MTC_RAOslot
    % * state vector (Ns1, Ns2, N_delayed, N_collided, Avg delay, n1, n2,
    
    %% Access Class Barring
    % Accessing media by using ACB factor
    q_var=rand(length(Index_request),1);

    % Delayed/barred devices (those that do not pass the ACB check) are recorded
    Index_delayed = []; Index_request_new = Index_request;
    n1_jj = 0; 
    for kk = 1:length(Index_request)
        % if a device has not yet been capable of passing the ACB check 
        if MTC_feat(Index_request(kk),4) == 0
            n1_jj = n1_jj + 1; % record it as such
            % additionally, if this device does not pass the ACB at
            % this slot...
            if q_var(kk) > P_ACB
                Index_request_new(kk) = 0; % delayed MTCDs are removed from this slot

                % if the MTCDs that do not pass the ACB check do not
                % exceed that max. allowed amount of  ACB check attempts
                if MTC_feat(Index_request(kk),3) + 1 < env5GConst.MaxWait + 1
                    % set them up to try to pass to ACB in subsequent
                    % RAO channels
                    Index_delayed = [Index_delayed; Index_request(kk)]; 
                end
            end     
        end
    end

    Index_request(Index_request_new == 0) = []; % remove delayed MTCDs
    clear kk Index_request_new;

    % Record the no. of devices that are competing for a preamble and
    % previously collided
    n2_jj = 0;
    for kk = 1:length(Index_request)
        if MTC_feat(Index_request(kk),4) > 0; n2_jj = n2_jj + 1; end
    end

    % Accessing media by using ACB factor
    T_barring=(0.7*ones(length(Index_delayed),1)+0.6*rand(length(Index_delayed),1))...
        .*(2.^MTC_feat(Index_delayed,3)); % Tbarring in sec.
    T_barring=ceil(T_barring/env5GConst.T_RAO); % Tbarring in normalized time    
    MTC_feat(Index_delayed,2)=MTC_feat(Index_delayed,2)+T_barring; % Delay
    MTC_RAOslot(Index_delayed)=MTC_RAOslot(Index_delayed)+T_barring';
    MTC_feat(Index_delayed,3)=MTC_feat(Index_delayed,3)+1; % Number of delays

    
    %% Preamble generation and designation     
    Preamble=randi(env5GConst.M,[length(Index_request),1]); 
    % M = no of available simultaneous communications
    [Preamble,Idx]=sort(Preamble);
    Index_request=Index_request(Idx);

    Prea_repe=zeros(size(Preamble,1)+1,1);
    Prea_expand = [0; Preamble];
    N_collisions = 0;
    for i1 = 2:length(Prea_expand)-1
        if Prea_expand(i1) == Prea_expand(i1+1)
            Prea_repe(i1:i1+1) = 1;
            if Prea_expand(i1) ~= Prea_expand(i1-1)
                N_collisions = N_collisions + 1;
            end
        end
    end
    Prea_repe = Prea_repe(2:end); 
    % remove the first element (additional element resulting from the 
    % expanded preamble version)

    % Record collided MTCDs as such
    Index_collided = Index_request(Prea_repe==1);

    Index_collided_new = Index_collided;
    for kk = 1:length(Index_collided)
        if MTC_feat(Index_collided(kk),3) + ... 
                MTC_feat(Index_collided(kk),4) + 1 > env5GConst.MaxWait + 1
            % ... the device is not allowed to access the system again
            Index_collided_new(kk) = 0; % MTCDs that collisioned over 
            % env5GConst.MaxWait times are removed from this slot
        end
    end
    Index_collided(Index_collided_new == 0) = []; % remove over-collisioned MTCDs
    clear kk Index_collided_new;

    T_BO=rand(length(Index_collided),1).*(2.^MTC_feat(Index_collided,3)); % T_BO in sec
    % Seria MTC_feat(Index_collided,4) i no pas amb 3
    T_BO=ceil(T_BO/env5GConst.T_RAO); % TBO in normaliced time    
    MTC_feat(Index_collided,2)=MTC_feat(Index_collided,2)+T_BO; % Delay
    MTC_feat(Index_collided,4)=MTC_feat(Index_collided,4)+1; % Number of collisions
    MTC_RAOslot(Index_collided)=MTC_RAOslot(Index_collided)+T_BO';
    
    % Indices of successfully connected MTCDs:
    Index_success = Index_request(Prea_repe == 0);

    % Successful transmissions reported:
    MTC_feat(Index_success,5) = 1; 

    Ns1_jj = 0; Ns2_jj = 0;
    for kk = 1:length(Index_success)
        % Record successfully connected MTCDs that did NOT previously
        % collide
        if MTC_feat(Index_success(kk),4) == 0
            Ns1_jj = Ns1_jj + 1;
        % Record successfully connected MTCDs that previously collided
        elseif MTC_feat(Index_success(kk),4) > 0
            Ns2_jj = Ns2_jj + 1;
        end
    end

    Avg_delay = 0;
    N_delayed = length(Index_delayed);
    if ~isempty(Index_success)
        Avg_delay = mean(MTC_feat(Index_success,2));
    end

    RAO_state = [Ns1_jj Ns2_jj N_delayed N_collisions Avg_delay n1_jj n2_jj];

end