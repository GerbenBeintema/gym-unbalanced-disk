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