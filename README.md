# gym-unbalanced-disk

This is the gym python environment for the unbalanced disk as seen below

![Setup](UnbalancedDisc1.jpg)

This repository includes 
1. benchmark datasets used for identification tasks.
2. **simulators** (both python and matlab)
3. Instruction and scripts to connect to the **experimental setup**

Note that the simulator is accurate model of the experimental setup for the provided model parameters. 

## 1. benchmark datasets used for identification tasks.

In `provided-measurement-datasets/` you can find two datasets which you can use for fitting and evaluating your models. However, for the design assignment we prefer (but not required) that you make you own measurements and that you motivate your experiment design choices in the report. 

## 2.1 Simulator python

### Installation

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

By using git clone you are able to change aspects of the observation space or action space easily.

### Use python simulator

```python
import gym, gym_unbalanced_disk, time

env = gym.make('unbalanced-disk-v0', dt=0.025, umax=3.) 
#env = gym_unbalanced_disk.UnbalancedDisk(dt=0.025, umax=3.) #alternative

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

Lastly, we also provide: 'unbalanced-disk-sincos-v0' or `gym_unbalanced_disk.UnbalancedDisk_sincos` which is the same environment but where angle is now expressed in sine and cos components. (useful for RL)

## 2.2 MATLAB Simulator

Download the `matlab-simulator` files and either use the function or the Simulink files. 

Note that The simulink model requires the Simulink 3D Animation toolbox to visualize the unbalanced disk setup (https://nl.mathworks.com/products/3d-animation.html). Note that the simulation files and figures should be added to the matlab path. It could be that the animation does not load well on first startup. Close the animation and open it again in case this happens.

# 3. Connecting to the experimental setup

# 3.1 Python connection

To use the experimental setup with the python environment you will need to follow steps.

1. Installation python simulator as shown before
2. Install the USB drivers using the instructions in `WindowsDcscUSB/README.txt`. 
3. Install MATLAB python engine. [See matlabworks](https://nl.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)
  * To reduce errors launch the "anaconda PowerShell prompt" with "run as admin" by right-clicking executable. 
  * Navigate to the MATLAB engine folder (e.g. `cd "C:\Program Files\MATLAB\R2021a\extern\engines\python"`) 
  * Use `python setup.py install`
  * Be sure that your python is compatible with you MATLAB version, [Table of compatible version](https://www.mathworks.com/content/dam/mathworks/mathworks-dot-com/support/sysreq/files/python-compatibility.pdf)
4. test connection by opening and running `examples-connect-to-exp/python-disk-test.ipynb`

Now use the following to create an environment with a connection to the system

```python
env = gym.make('unbalanced-disk-exp-v0', dt=0.025, umax=3.) #both are equivilent
env = gym_unbalanced_disk.UnbalancedDisk(dt=0.025, umax=3.) #both are equivilent
```

# 3.2 Matlab connection

To use the experimental setup you will need to do the following things.

1. download `examples-connect-to-exp`
2. Install the USB drivers using the instructions in `WindowsDcscUSB/README.txt`. 
3. test connection by running `examples-connect-to-exp/matlab_disk_test.m`
