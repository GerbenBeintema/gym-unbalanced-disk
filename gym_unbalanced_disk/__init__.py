from gymnasium.envs.registration import register
from gym_unbalanced_disk.envs.UnbalancedDisk import UnbalancedDisk, UnbalancedDisk_sincos
from gym_unbalanced_disk.envs.UnbalancedDiskExpMatlab import UnbalancedDisk_exp_matlab, UnbalancedDisk_exp_matlab_sincos
from gym_unbalanced_disk.envs.UnbalancedDiskExpPython import UnbalancedDisk_exp_python, UnbalancedDisk_exp_python_sincos

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
    id='unbalanced-disk-exp-matlab-v0',
    entry_point='gym_unbalanced_disk.envs:UnbalancedDisk_exp_matlab',
    max_episode_steps=300
)

register(
    id='unbalanced-disk-exp-matlab-sincos-v0',
    entry_point='gym_unbalanced_disk.envs:UnbalancedDisk_exp_matlab_sincos',
    max_episode_steps=300
)

register(
    id='unbalanced-disk-exp-python-v0',
    entry_point='gym_unbalanced_disk.envs:UnbalancedDisk_exp_python',
    max_episode_steps=300
)

register(
    id='unbalanced-disk-exp-python-sincos-v0',
    entry_point='gym_unbalanced_disk.envs:UnbalancedDisk_exp_python_sincos',
    max_episode_steps=300
)