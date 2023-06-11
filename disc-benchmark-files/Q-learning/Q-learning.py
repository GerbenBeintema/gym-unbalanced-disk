#%%
import gym, gym_unbalanced_disk, time
import numpy as np
import os
from UnbalancedDisk import *

import matplotlib.pyplot as plt



env = UnbalancedDisk() #alternative


# env = gym.make('unbalanced-disk-v0', dt=0.025, umax=3.) 

#%%

def argmax(a):
    #random argmax
    a = np.array(a)
    return np.random.choice(np.arange(len(a),dtype=int)[a==np.max(a)])


def QlearnGrid(env,Qmat = None, alpha=0.1, epsilon=0.1, gamma=1.0, nsteps=100000, epsilon_decay=True, visualize=False):
    #init Q:
    
    obs_list = []
    reward_list = []
    if Qmat is None:
        Qmat = np.zeros(([360,30,env.action_space.n])) # i chose the amount of states for accel randomly
    
    
    obs = env.reset()
    for z in range(nsteps):
        if epsilon_decay:
            epsilon = 1- (z / nsteps)

        if np.random.uniform()<epsilon: 
            action = env.action_space.sample() 
    
        else: 
            action = argmax(Qmat[round(obs[0]),round(obs[1]),:])
        obs_new, reward, done, _ = env.step(action) 
        
        if done:
            TD = reward - Qmat[obs[0],obs[1],action] 
            Qmat[round(obs[0]),round(obs[1]),action] += alpha*TD 
            obs = env.reset()
        else:
            MaxQ = max(Qmat[round(obs_new[0]),round(obs_new[1]),action_next] for action_next in range(env.action_space.n))
            TD = reward + gamma*MaxQ - Qmat[round(obs[0]),round(obs[1]),action] 
            Qmat[round(obs[0]),round(obs[1]),action] += alpha*TD 
            obs = obs_new 
        
        obs_list.append(obs)
        reward_list.append(reward)
        if visualize:
            env.render()
            time.sleep(1/24)
    
    return Qmat, obs_list, reward_list



# %%
Qmat, obs_list, reward_list = QlearnGrid(env, alpha=0.1, epsilon=0.1, gamma=1.0, nsteps=300000, epsilon_decay=False, visualize=False)
#%%
visualize_range = -1000
plt.plot((np.array(obs_list)[:,0]), label="angle" )
plt.plot(np.array(reward_list), label="reward" , alpha=0.5)
plt.yscale("symlog")
plt.legend()
plt.show()
#%%
np.save("Qmat.npy",Qmat)
# %%
#use the Q matrix calculated to visualize it
np.load("Qmat.npy")
#%%
Qmat, obs_list, reward_list = QlearnGrid(env, Qmat=Qmat, alpha=0.1, epsilon=0, gamma=1.0, nsteps=5000, epsilon_decay=False, visualize=True)
# %%
