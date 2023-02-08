clear all
close all
clc

state = init_state_disk();

dt = 0.025;
for i = 1:101
    thetas(i) = state.theta;
    u = 3;
    state = unbalanced_disk(state, u, dt);
end
plot(thetas)
title(num2str(max(thetas)))
