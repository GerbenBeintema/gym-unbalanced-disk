# gym-unbalanced-disk

This is the gym python environment for the unbalanced disk as seen below

![Setup](UnbalancedDisc1.jpg)

This repository includes 
1. benchmark datasets used for identification tasks.
2. **simulators** (both python and matlab)
3. Instruction and scripts to connect to the **experimental setup**

Note that the simulator is accurate model of the experimental setup for the provided model parameters. 

## 1. benchmark datasets used for identification tasks.

In `disc-benchmark-files/` you can find

* `training-data.[csv,mat,npz]` which contains the data for your system identification task – you can partition this data into training, validation and test.
* `test-prediction-submission-file.[csv,mat,npz]` contains the (past) input and output data required to perform a 1 step-ahead prediction. This also illustrates the file format that we expect for the prediction task submission (by replacing the zero entries of the y[k-0] output with your estimate).
* `test-simulation-submission-file.[csv,mat,npz]` contains the input sequence and the first 50 output values of the system. The later output samples are replaced by zeros. This data should be used to simulate the remainder of the outputs of the system. This also illustrates the file format that we expect for the simulation task submission (by replacing the zero entries of the output with your estimate).
* `example-prediction-solution.py` an example file which shows a simple linear ARX prediction solution using the datasets provided.
* `example-simulation-solution.py` an example file which shows a simple linear ARX simulation solution using the datasets provided.
* `submission-file-checker.py` is run as `python submission-file-checker.py submitted-file solution-file` to compute the prediction/simulation errors. You can also run this file to check if your file has the appropriate format by running `python submission-file-checker.py submitted-file test-prediction-submission.npz` which successfully ends without an error if the submitted-file has the correct format
 
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
env = gym.make('unbalanced-disk-exp-v0', dt=0.025, umax=3.) #both are equivilent (this one has a time limit build in)
env = gym_unbalanced_disk.UnbalancedDisk(dt=0.025, umax=3.) #both are equivilent
```

# 3.2 Matlab connection

To use the experimental setup you will need to do the following things.

1. download `examples-connect-to-exp`
2. Install the USB drivers using the instructions in `WindowsDcscUSB/README.txt`. 
3. test connection by running `examples-connect-to-exp/matlab_disk_test.m`
