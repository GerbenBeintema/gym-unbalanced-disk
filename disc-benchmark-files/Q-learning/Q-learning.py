#%%
import gym, gym_unbalanced_disk, time
import numpy as np
import os
from UnbalancedDisk import *
import matplotlib.pyplot as plt

env = UnbalancedDisk_sincos_cus() #alternative


# env = gym.make('unbalanced-disk-v0', dt=0.025, umax=3.) 

#%%

def argmax(a):
    #random argmax
    a = np.array(a)
    return np.random.choice(np.arange(len(a),dtype=int)[a==np.max(a)])


def QlearnGrid(env,Qmat = None, alpha=0.1, epsilon=0.1, gamma=1.0, nsteps=100000, epsilon_decay=True, visualize=False):
    #init Q:
    factor = 100
    obs_list = np.zeros((nsteps,2))
    reward_list = np.zeros((nsteps,1))
    highest_reward = -1000
    if Qmat is None:
        Qmat = np.zeros(([360,30,env.action_space.n])) # i chose the amount of states for accel randomly
    cur_nsteps = 0
    
    obs_cos = env.reset()
    obs = np.array([np.arctan2(obs_cos[0], obs_cos[1])*factor, obs_cos[2]])
    for z in range(nsteps):
        if epsilon_decay:
            epsilon = 1- (z / nsteps)

        if np.random.uniform()<epsilon: 
            action = env.action_space.sample() 
    
        else: 
            action = argmax(Qmat[round(obs[0]),round(obs[1]),:])
        obs_new_cos, reward, done, _ = env.step(action) 
        obs_new = np.array([np.arctan2(obs_new_cos[0], obs_new_cos[1])*factor, obs_new_cos[2]])
        # make sure that the angle is between 0 and 2pi so it doesn't attempt to 
        # learn outside of the space
        if done:
            TD = reward - Qmat[obs[0],obs[1],action] 
            Qmat[round(obs[0]),round(obs[1]),action] += alpha*TD 
            obs = env.reset()
        else:
            MaxQ = max(Qmat[round(obs_new[0]),round(obs_new[1]),action_next] for action_next in range(env.action_space.n))
            TD = reward + gamma*MaxQ - Qmat[round(obs[0]),round(obs[1]),action] 
            Qmat[round(obs[0]),round(obs[1]),action] += alpha*TD 
            obs = obs_new 
        highest_reward = max(highest_reward,reward)
        
        obs_list[z] = obs
        reward_list[z] = reward
        if visualize:
            env.render()
            time.sleep(1/24)
            print(f"{z}/{nsteps} - highest reward: {highest_reward:.4f} cur reward ={reward:.4f} ", end='\r')
            
        if z == cur_nsteps + 1000:
            print(f"{z}/{nsteps} -cur_angle {(obs[0]/factor):3f}-cur_vel {obs[1]:.3f}- highest reward: {highest_reward:.4f} cur reward ={reward:.4f} ", end='\r')
            
            cur_nsteps = z
    
    return Qmat, obs_list, reward_list



# %%
Qmat, obs_list, reward_list = QlearnGrid(env, alpha=0.1, epsilon=0.2, gamma=1.0, nsteps=100000, epsilon_decay=False, visualize=False)
#%%
visualize_range = -1000
plt.plot(obs_list[:,0], label="angle" ,alpha=0.5)
plt.plot((obs_list[:,1]), label="velocity" ,alpha=0.8)
plt.plot((reward_list), label="reward" , alpha=0.5)
plt.yscale("symlog")
plt.legend()
plt.show()
#%%
np.save("Qmat.npy",Qmat)
#%%
Qmat = np.load("Qmat.npy")
#%%
#use the Q matrix calculated to visualize it
Qmat, obs_list, reward_list = QlearnGrid(env, Qmat=Qmat, alpha=0.1, epsilon=0.2, gamma=1.0, nsteps=10000, epsilon_decay=False, visualize=True)

# %%
