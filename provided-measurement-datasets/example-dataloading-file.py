import numpy as np
from matplotlib import pyplot as plt

#you can use both dataset 1 and dataset 2 but you need to make the train, val, test splits
out = np.load('./disk-measurement-dataset-2.npz') 
u = out['u'] #inputs
th = out['th'] #outputs
t = out['t'] #time vector

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(t,u)
plt.xlabel('time (sec)')
plt.ylabel('input (V)')
plt.subplot(1,2,2)
plt.plot(t,th)
plt.xlabel('time (sec)')
plt.ylabel('$\\theta$ (rad)')
plt.tight_layout()
plt.show()