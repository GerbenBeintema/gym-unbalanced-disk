function [O, S] = fugiboard(C, OS, CB, DO, DACS, OTHERS) %#ok
%FUGIBOARD General interface to DCSC FUGI board based experimental setups
%
%   In the following desctiptions is O an object containing the settings of
%   a setup, is H an object returned by the 'GetParams' or 'Open' commands, 
%   is S 0 (zero) when the command completed succesfully, is R an object
%   with the values of the sensors in the setup and is A an array with the
%   values of the sensors.
%
%   fugiboard('Help', 'SETUPn')
%            Displays a short message with the parameters for the Read and
%            Write commands for SETUP n. SETUP can be MOPS, BALLPLATE,
%            HUBOAT, or PENDULUM.
%   fugiboard('Help', H)
%            Displays a short message with the parameters for the Read and
%            Write commands for the setup identified by object H. SETUP can
%            be MOPS, BALLPLATE, HUBOAT, or PENDULUM.
%
%   [O, S] = fugiboard('GetParams', 'SETUPn')
%            Returns the current settings for SETUP n in object O and the
%            command status in S. SETUP can be MOPS, BALLPLATE, HUBOAT, or
%            PENDULUM.
%   [O, S] = fugiboard('GetParams', H)
%            Returns the current settings for the setup identified by
%            object H in object O and the command status in S. H must be
%            returned by the GetParams or Open commands.
%
%   S = fugiboard('SetParams', H)
%            Modifies the settings of the setup identified by object H.
%            Most settings are copied from object H, but the device field
%            can't be modified and the Error fields will be reset to zero.
%            Returns the command status in S.
%
%   [O, S] = fugiboard('Open', 'SETUPn')
%            Returns the current settings for SETUP n and opens a
%            connection to that setup, using the current settings. The
%            object O should be used as object H in other calls to
%            fugiboard. SETUP can be MOPS, BALLPLATE, HUBOAT, or
%            PENDULUM.Returns the command status in S.
%   [O, S] = fugiboard('Open', H)
%            Modifies the settings of the setup identified by object H, as
%            explained for the SetParams command. After changing the
%            settings a connection to the setup will be opened. The object
%            O should be used as object H in other calls to fugiboard.
%            Returns the command status in S
%
%   [A, S] = fugiboard('Read', H)
%            Reads the values from the sensors in the setup, scales them
%            and returns them in array A. Return the commmand status in S.
%            The meaning of the elements of A is given by the 'Help'
%            action. The status of the setup, returned in A(1) may report
%            an error, even when the command itself succeeded. 
%
%            The status of the setup, A(1), is the sum of:
%                1       encoder index pulse seen since last sample
%                2       watchdog timed out since last command
%                16      current ADC timed out
%                32      inclino meter X ADC timed out
%                64      inclino meter ADC timed out
%
%   S = fugiboard('Write', H, C, E,....)
%            Sends a command to the actuators in the setup identified by
%            object H. The command parameters are translated to raw binary
%            data before transmission. The command paramerers are given by
%            the 'Help' action. H is only used to identify the setup, the
%            settings in H are not used! Returns the command status in S.
%
%            C: command
%                0:      set outputs E, D1 and D2 (normal operation)
%                1:      reset position counters
%                2:      reset position counters on next rising edge of the 
%                        index pulse from the incremental encoder
%                3:      reset position counter immediately, do not count
%                        until the next rising edge of the index pulse from the
%                        incremental encoder
%                4:      return the model and version numbers in stead of
%                        the status in 'Read' commands
%                Remember to set C to 0 (zero) again, else the setup will keep
%                repeating the command forever.
%            E: digital outputs
%                0:      disable the analogue outputs
%                1:      enable the analogue outputs
%                The setup uses a relay to enable the outputs, so this value
%                should not be changed very frequently.
%
%   S = fugiboard('Close', H)
%            Closes the connection to to the setup identified by object H.
%            Returns the command status in S.
%
%   fugiboard('CloseAll')
%            Closes all connections to setups.

% The following comment, MATLAB compiler pragma, is necessary to avoid compiling 
% this M-file instead of linking against the MEX-file.  Don't remove.
%# mex

error('C-MEX function not found');

% [EOF] fugiboard.m
