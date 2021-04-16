# gym-unbalanced-disk

This is the gym python environment for the unbalanced disk as seen below

![Setup](UnbalancedDisc1.jpg)

This includes both a simulated and an experimental setup. 

## Installation simulation

Use command prompt or equivalent and enter

```
python -m pip install git+git://github.com/GerbenBeintema/gym-unbalanced-disk@master
```

or download the repository and install using

```
git clone git@github.com:upb-lea/gym-unbalanced-disk.git #(or use manual download on the github page)
cd gym-unbalanced-disk
pip install -e .
```

## Additional Experimental setup install

To use the experimental setup you will need to do the following things.

1. Install the usb drivers using the instructions in `WindowsDcscUSB/`. 
2. Install MATLAB python engine. [See matlabworks](https://nl.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)

## Basic use of simulation

```
import gym, gym_unbalanced_disk, time

env = gym.make('unbalanced-disk-v0', dt=0.025, umax=4.) 
#or use gym_unbalanced_disk.UnbalancedDisk(dt=0.025, umax=4.)

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

## Environments

### Simulations

'unbalanced-disk-v0' or `gym_unbalanced_disk.UnbalancedDisk` which is the simulation

'unbalanced-disk-sincos-v0' or `gym_unbalanced_disk.UnbalancedDisk_sincos` same as above where the angle is now expressed in sine and cos components. (useful for RL)

### Experimental setups.

'unbalanced-disk-exp-v0' or `gym_unbalanced_disk.UnbalancedDisk_exp` which is the experiment

'unbalanced-disk-exp-sincos-v0' or `gym_unbalanced_disk.UnbalancedDisk_exp_sincos` same as above where the angle is now expressed in sine and cos components. (useful for RL)
