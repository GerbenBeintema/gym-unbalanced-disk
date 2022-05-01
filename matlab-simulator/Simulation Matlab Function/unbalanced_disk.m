%% unballanced disk: function description
function [new_state] = unbalanced_disk(state, u, dt)

    u = min(max(u,-3),3);
    %%############# start do not edit  ################
    g = 9.80155078791343;
    J = 0.000244210523960356;
    Km = 10.5081817407479;
    I = 0.0410772235841364;
    M = 0.0761844495320390;
    tau = 0.397973147009910;
    %%############# end do not edit ###################
    
    
    y0 = [state.theta, state.omega];
    f = @(t, y) [y(2); -M*g*I/J*sin(y(1)) - 1/tau*y(2) + Km/tau*u];
    [t,y] = ode45(f, [0, dt], y0);
    s = size(y);
    new_state.theta = y(s(1),1);
    new_state.omega = y(s(1),2);
    
    % new_state.theta = y(-1)
