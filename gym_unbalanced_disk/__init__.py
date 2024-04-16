from gym.envs.registration import register
from gym_unbalanced_disk.envs.UnbalancedDisk import UnbalancedDisk, UnbalancedDisk_sincos
from gym_unbalanced_disk.envs.UnbalancedDiskExp import UnbalancedDisk_exp_matlab, UnbalancedDisk_exp_matlab_sincos

register(
    id='unbalanced-disk-v0',
    entry_point='gym_unbalanced_disk.envs:UnbalancedDisk',
    max_episode_steps=300
)

register(
    id='unbalanced-disk-sincos-v0',
    entry_point='gym_unbalanced_disk.envs:UnbalancedDisk_sincos',
    max_episode_steps=300
)

register(
    id='unbalanced-disk-exp-v0',
    entry_point='gym_unbalanced_disk.envs:UnbalancedDisk_exp_matlab',
    max_episode_steps=300
)

register(
    id='unbalanced-disk-exp-sincos-v0',
    entry_point='gym_unbalanced_disk.envs:UnbalancedDisk_exp_matlab_sincos',
    max_episode_steps=300
)