%% unballanced disk: function description
function [new_state] = unbalanced_disk(state, u, dt)

    
    %%############# start do not edit  ################
    u = min(max(u,-3),3);
    % g = 9.80155078791343;
    % J = 0.000244210523960356;
    % Km = 10.5081817407479;
    % I = 0.0410772235841364;
    % M = 0.0761844495320390;
    % tau = 0.397973147009910;

    omega0 = 11.339846957335382;
    delta_th = 0.;
    gamma = 1.3328339309394384;
    Ku = 28.136158407237073;
    Fc = 6.062729509386865;
    coulomb_omega = 0.001;
    
    y0 = [state.theta, state.omega];
    % f = @(t, y) [y(2); -M*g*I/J*sin(y(1)) - 1/tau*y(2) + Km/tau*u];
    f = @(t, y) [y(2); -omega0^2*sin(y(1) + delta_th) - (gamma*y(2) + Fc*tanh(y(2)/coulomb_omega)) + Ku*u];
    [t,y] = ode45(f, [0, dt], y0);
    s = size(y);
    new_state.theta = y(s(1),1);
    new_state.omega = y(s(1),2);
    %%############# end do not edit ###################
end