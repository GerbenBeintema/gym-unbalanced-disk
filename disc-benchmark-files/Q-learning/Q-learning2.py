#%%
from collections import defaultdict
import gym, time
import numpy as np
import os
from UnbalancedDisk2 import *
import matplotlib.pyplot as plt 

#%%
def argmax(a):
    a = np.array(a)
    return np.random.choice(np.arange(a.shape[0],dtype=int)[a==np.max(a)])


from collections import defaultdict
def Qlearn(env,Qmat, alpha=0.1, epsilon=0.1, gamma=1.0, nsteps=10000,visualize=False, epsilon_decay=False):
     #any new argument be set to zero
    obs_list = np.zeros((nsteps,2))
    reward_list = np.zeros((nsteps,1))
    highest_reward = -1000
    obs_temp = env.reset()
    cur_nsteps = 0
    obs = np.array([round(np.arctan2(obs_temp[0], obs_temp[1]),1),round(obs_temp[2],1)])
    for z in range(nsteps):
        if epsilon_decay:
            epsilon = 1- (z / nsteps)

        if np.random.uniform()<epsilon:
            action = env.action_space.sample()
        else:
            action = argmax([Qmat[obs[0],obs[1],a] for a in range(env.action_space.n)])

        obs_new_temp, reward, done, _ = env.step(action)
        obs_new = np.array([round(np.arctan2(obs_new_temp[0], obs_new_temp[1]),1), round(obs_new_temp[2],1)])
        if done:
            TD = reward - Qmat[obs[0],obs[1],action]
            Qmat[obs[0],obs[1],action] += alpha*TD
            obs_temp = env.reset()
        else:
            Qmax = max(Qmat[obs_new[0], obs_new[1], action_next] for action_next in range(env.action_space.n))
            TD = reward + gamma*Qmax - Qmat[obs[0],obs[1],action]
            Qmat[obs[0],obs[1],action] += alpha*TD
            obs_temp = obs_new
        if z == cur_nsteps + 1000:
            print(f"{z}/{nsteps} -cur_angle {(obs[0]):3f}-cur_vel {obs[1]:.3f}- highest reward: {highest_reward:.4f} cur reward ={reward:.4f} ", end='\r')
            cur_nsteps = z
        highest_reward = max(highest_reward, reward)
        obs_list[z] = obs_temp
        reward_list[z] = reward
        obs = obs_new
        if visualize:
            env.render()
            time.sleep(1/60)
            print(f"{z}/{nsteps} -cur_angle {(obs[0]):3f}-cur_vel {obs[1]:.3f}- highest reward: {highest_reward:.4f} cur reward ={reward:.4f} ", end='\r')

    return Qmat, obs_list, reward_list
#%%
Qmat = defaultdict(lambda: float(0))
#%%
env = UnbalancedDisk_sincos_cus()
#%%

Qmat,obs_list, reward_list = Qlearn(env,Qmat, alpha=0.2, epsilon=0.2, gamma=0.99, nsteps=400_000, epsilon_decay=False)

# %%

visualize_range = -1000
plt.plot(obs_list[:,0], label="angle" ,alpha=0.5)
plt.plot((obs_list[:,1]), label="velocity" ,alpha=0.8)
plt.plot((reward_list), label="reward" , alpha=0.5)
plt.yscale("symlog")
plt.legend()
plt.show()

#%%

Qmat,obs_list, reward_list = Qlearn(env,Qmat, alpha=0.1, epsilon=0, gamma=0.99, nsteps=300,visualize=True)

# %%
import pickle
file_path = r"Good_Qmat.pkl"
with open(file_path, "wb") as file:
    pickle.dump(Qmat, file)
# %%
