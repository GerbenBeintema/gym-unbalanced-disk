# gym-unbalanced-disk

Installation

```
python -m pip install git+git://github.com/GerbenBeintema/gym-unbalanced-disk@master
```

or

```
git clone git@github.com:upb-lea/gym-electric-motor.git 
#(or use download github page)
cd gym-electric-motor
pip install -e .
```


basic use

```
import gym, gym_unbalanced_disk, time


env = gym.make('unbalanced-disk-v0')

obs = env.reset()
for i in range(200):
    obs, reward, done, info = env.step(env.action_space.sample())
    print(obs,reward)
    env.render()
    time.sleep(1/24)
    if done:
        obs = env.reset()
env.close()
```
