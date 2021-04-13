from gym.envs.registration import register
from gym_unbalanced_disk.envs.UnbalancedDisk import UnbalancedDisk, UnbalancedDisk_th

register(
    id='unbalanced-disk-v0',
    entry_point='gym_unbalanced_disk.envs:UnbalancedDisk',
    max_episode_steps=300
)