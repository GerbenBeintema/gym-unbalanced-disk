% Find the hex file to download to the setup (only required for Mac)
if strcmp(computer, 'MACI64') && strcmp(getenv('DCSCUSB_HEX_FILE'), '')
    setenv('DCSCUSB_HEX_FILE', which('fx2-slave-fifo.hex'));
end;

clear x*;
fugiboard('CloseAll');
h=fugiboard('Open', 'mops1');
h.WatchdogTimeout = 2;
fugiboard('SetParams', h);
fugiboard('Write', h, 0, 0, [0 0]);  % dummy write to sync interface board
fugiboard('Write', h, 4+1, 1, [0 0]);  % get version, reset position, activate relay
data = fugiboard('Read', h);
model = bitshift(data(1), -4);
version = bitand(data(1), 15);
disp(sprintf('FPGA setup %d,  version %d', model, version));
fugiboard('Write', h, 0, 1, [0 0]);  % end reset

pause(0.5); % give relay some time to act
steps = 2000;
xstat = zeros(1,steps);
xreltime = zeros(1,steps);
xpos = zeros(1,steps);
xspd = zeros(1,steps);
xcurr = zeros(1,steps);
xvolt = zeros(1,steps);
xextn = zeros(1,steps);
xdigin = zeros(1,steps);
xcspd = zeros(1,steps);
tic;
bt = toc;
for X=1:steps
    if (X > 100000)
        fugiboard('Write', h, 0, 1, [5.0 0.0]);
    else
        fugiboard('Write', h, 0, 1, [7.5 0.0]);
    end
    data = fugiboard('Read', h);
    xstat(X) = data(1);
    xreltime(X) = data(2);
    xpos(X) = data(3);
    xspd(X) = data(4);
    xcurr(X) = data(5);
    xvolt(X) = data(6);
    xextn(X) = data(7);
    xdigin(X) = data(8);
    xcspd(X) = data(9);
    t = bt + (0.001 * X);
    %t = toc + 0.005;
    while (toc < t); end;
end
toc;
fugiboard('Write', h, 0, 0, [0.0 0.0]);
figure(1); stairs(xpos); ylabel('Position');
figure(2); stairs(xspd); ylabel('Speed');
figure(3); stairs(xcurr); ylabel('Current');
figure(4); stairs(xvolt); ylabel('Volt'); axis([0 steps 0 10]);