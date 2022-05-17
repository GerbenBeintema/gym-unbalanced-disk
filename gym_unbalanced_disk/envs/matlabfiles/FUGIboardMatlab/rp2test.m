% Find the hex file to download to the setup (only required for Mac)
if strcmp(computer, 'MACI64') && strcmp(getenv('DCSCUSB_HEX_FILE'), '')
    setenv('DCSCUSB_HEX_FILE', which('fx2-slave-fifo.hex'));
end;

clear x*;
fugiboard('CloseAll');
h=fugiboard('Open', 'pendulum1');
h.WatchdogTimeout = 1;
fugiboard('SetParams', h);
fugiboard('Write', h, 0, 0, [0 0]);    % dummy write to sync interface board
fugiboard('Write', h, 4+1, 1, [0 0]);  % get version, reset position, activate relay
data = fugiboard('Read', h);
model = bitshift(data(1), -4);
version = bitand(data(1), 15);
disp(sprintf('FPGA setup %d,  version %d', model, version));
fugiboard('Write', h, 0, 1, [0 0]);  % end reset

pause(0.1); % give relay some time to act
steps = 2000;
xstat = zeros(1,steps);
xreltime = zeros(1,steps);
xpos1 = zeros(1,steps);
xpos2 = zeros(1,steps);
xcurr = zeros(1,steps);
xbeam = zeros(1,steps);
xpend = zeros(1,steps);
xdigin = zeros(1,steps);
tic;
bt = toc;
for X=1:steps
    if (X > 1000)
        fugiboard('Write', h, 0, 1, [10.0 0.0]);
    else
        fugiboard('Write', h, 0, 7, [-10.0 0.0]);
    end
    data = fugiboard('Read', h);
    xstat(X) = data(1);
    xreltime(X) = data(2);
    xpos1(X) = data(3);
    xpos2(X) = data(4);
    xcurr(X) = data(5);
    xbeam(X) = data(6);
    xpend(X) = data(7);
    xdigin(X) = data(8);
    t = bt + (0.001 * X);
    %t = toc + 0.005;
    %figure(3); stairs([xbeam; xpend]'); ylabel('Beam, Pendulum');

    while (toc < t); end;
end
toc;
fugiboard('Write', h, 0, 0, [0.0 0.0]);
figure(1); stairs([xpos1; xpos2]'); ylabel('Position1, Position2');
figure(2); stairs(xcurr); ylabel('Current');
figure(3); stairs([xbeam; xpend]'); ylabel('Beam, Pendulum');
figure(4); stairs(xdigin); ylabel('DigIN');
