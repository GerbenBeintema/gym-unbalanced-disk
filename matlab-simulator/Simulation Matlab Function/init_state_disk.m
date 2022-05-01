function [state] = init_state_disk()
%INIT_STATE_DISK Summary of this function goes here
%   Detailed explanation goes here
state.theta = randn()*0.001;
state.omega = randn()*0.001;
end

