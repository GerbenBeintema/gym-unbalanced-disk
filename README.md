# gym-unbalanced-disk

This is the gym python environment for the unbalanced disk as seen below

![Setup](UnbalancedDisc1.jpg)

This includes both a **simulator** and an instruction and scripts to connect to the **experimental setup**. 

Note that the simulator is accurate model of the experimental setup for the provided model parameters. 

# Provided data

In `provided-measurement-datasets/` you can find two datasets which you can use for fitting and evaluating your models. However, for the design assignment we prefer (but not required) that you make you own measurements and that you motivate your experiment design choices in the report. 

# Simulator

## Installation python Simulator

Use any terminal or equivalent and enter

```
python -m pip install git+https://github.com/GerbenBeintema/gym-unbalanced-disk@master
```

or download the repository and install using

```
git clone https://github.com/GerbenBeintema/gym-unbalanced-disk.git #(or use manual download on the github page)
cd gym-unbalanced-disk
pip install -e .
```

### Basic use of python simulator

```
import gym, gym_unbalanced_disk, time

env = gym.make('unbalanced-disk-v0', dt=0.025, umax=4.) 
#env = gym_unbalanced_disk.UnbalancedDisk(dt=0.025, umax=4.) #alternative

obs = env.reset()
try:
    for i in range(200):
        obs, reward, done, info = env.step(env.action_space.sample()) #random action
        print(obs,reward)
        env.render()
        time.sleep(1/24)
        if done:
            obs = env.reset()
finally: #this will always run
    env.close()
```

## Installation MATLAB Simulator

Download the `matlab-simulator` files and either use the function or the Simulink files. 

# Experimental Setup

## Requirements to be able to connect to the experimental setup

To use the experimental setup you will need to do the following things.

1. Install the USB drivers using the instructions in `WindowsDcscUSB/README.txt`. 
2. Install MATLAB python engine. [See matlabworks](https://nl.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)
  * To reduce errors launch the "anaconda PowerShell prompt" with "run as admin" by right-clicking executable. 
  * Navigate to the MATLAB engine folder (e.g. `cd "C:\Program Files\MATLAB\R2021a\extern\engines\python"`) 
  * Use `python setup.py install`
  * Be sure that your python is compatible with you MATLAB version, [Table of compatible version](https://www.mathworks.com/content/dam/mathworks/mathworks-dot-com/support/sysreq/files/python-compatibility.pdf)

## Environments

### Simulations

'unbalanced-disk-v0' or `gym_unbalanced_disk.UnbalancedDisk` which is the simulation

'unbalanced-disk-sincos-v0' or `gym_unbalanced_disk.UnbalancedDisk_sincos` same as above where the angle is now expressed in sine and cos components. (useful for RL)

### Experimental setups.

'unbalanced-disk-exp-v0' or `gym_unbalanced_disk.UnbalancedDisk_exp` which is the experiment

'unbalanced-disk-exp-sincos-v0' or `gym_unbalanced_disk.UnbalancedDisk_exp_sincos` same as above where the angle is now expressed in sine and cos components. (useful for RL)
