DcscUSB driver

The DcscUSB driver is necessary to connect FPGA based laboratory setups designed 
by DCSC technical staff to a PC. Currently the driver supports the 64 bits versions of
Windows 7, 8.1 and 10 (32 bits should work too but is untested).

To be able to install this driver you need to allow for the installation 
of the unsigned driver in the following steps.

Installation for Windows 10/11:
-------------------------------

A disable driver signature enforcement:
1) Connect a FPGA based setup, the driver installation will silently fail.
2) Open the Start Menu, type "Bitlocker" and start it if present, otherwise continue to step 4.
3) Click on "Back up your recovery key" and save on other device/write down your 48 digit recovery key.
4) Hold shift while selecting restart to open Advcanced startup
5) Click on the Start Menu button and type "recovery options".
6) Click on Troubleshoot.
7) Click on Advanced options.
8) Click on Startup Settings. (You might need to select "more options")
9) Click on Restart.
10) Wait while the PC restarts and gets to the Startup settings page. 
11) Enter the bitlocker code noted down in step 3 if asked for.
12) Then press key 7 to select "7) Disable driver signature enforcement".
13) Login again and locate the WindowsDcscUSB folder with the drivers. (cannot be contained in a zip)
14) Open the Start Menu, type "device manager" and start it.
15) Right-click on the name of the setup (sometimes just called Unknown device) 
    under "Other devices" and select Update driver software....
16) Choose to Browse for driver software on your computer and point it to the Win10 
    sub-directory in the directory from step 11.
17) Allow the installation of the unsigned driver. If the installation fails because 
    a hash is missing, driver signature enforcment is still/again on. Retry the 
    procedure from step 1. If the installation fails again, seek help.

 
Installation for Windows 8.1:
-----------------------------

Note: If driver signature enforcement is already disabled, start with step 12.
1) Connect a FPGA based setup, the driver installation will silently fail.
2) Open the charms menu (top right corner) and click on Settings.
3) Click on Change PC settings.
4) Click on Update and recovery.
5) Click on Recovery
6) Click on Restart now under Advanced start-up.
7) Click on Troubleshoot.
8) Click on Advanced options,
9) Click on Start-up Settings
10) Click on Restart
11) Wait while the PC restarts and gets to the Startup settings page. (don't press anything for this step)
12) Then press key 7 to select "7) Disable driver signature enforcement".
13) Login again and unpack the zip file in a directory.
14) Open the charms menu, search for "device manager" and start it.
15) Right-click on the name of the setup (sometimes just called Unknown device) under 
    Other devices and select Update driver software....
16) Choose Browse for driver software on your computer and point it to the Win81 subdirectory
    from in the directory from step 12.
17) Allow the installation of the unsigned driver. If the installation fails because a
    hash is missing, driver signature enforcment is still/again on. Retry the procedure
    from step 1. If the installation fails again, seek help.

 
Installation for Windows 7:
---------------------------

1) Unpack the zip file in a directory.
2) Connect a FPGA based setup, the driver installation will fail.
3) Open the Windows Control Panel, click on System and select the Device Manager. 
4) Right-click on the name of the setup (sometimes just called Unknown device) under 
   Other devices and select Update driver software....
5) Choose Browse for driver software on your computer and point it to the directory from 
    step 1.
6) Allow the installation of the unsigned driver.
