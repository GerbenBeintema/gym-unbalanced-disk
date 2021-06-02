
import gym
from gym import spaces
import numpy as np
from scipy.integrate import solve_ivp
from os import path

class UnbalancedDisk(gym.Env):
    '''
    UnbalancedDisk
    th =            
                  +-pi
                    |
           pi/2   ----- -pi/2
                    |
                    0  = starting location
    '''
    def __init__(self, umax=3., dt = 0.025):
        ############# start do not edit  ################
        self.g = 9.80155078791343
        self.J = 0.000244210523960356
        self.Km = 10.5081817407479
        self.I = 0.0410772235841364
        self.M = 0.0761844495320390
        self.tau = 0.397973147009910
        ############# end do not edit ###################

        self.umax = umax
        self.dt = dt #time step
 

        # change anything here (compilable with the exercise instructions)
        self.action_space = spaces.Box(low=-umax,high=umax,shape=tuple()) #continues
        # self.action_space = spaces.Discrete(2) #discrete
        low = [-float('inf'),-40] 
        high = [float('inf'),40]
        self.observation_space = spaces.Box(low=np.array(low,dtype=np.float32),high=np.array(high,dtype=np.float32),shape=(2,))

        self.reward_fun = lambda self: np.exp(-(self.th%(2*np.pi)-np.pi)**2/(2*(np.pi/7)**2)) #example reward function, change this!
        
        self.viewer = None
        self.u = 0 #for visual
        self.reset()

    def step(self, action):
        #convert action to u
        self.u = action #continues
        # self.u = [-3,-1,0,1,3][action] #discrate
        # self.u = [-3,3][action] #discrate

        ##### Start Do not edit ######
        self.u = np.clip(self.u,-self.umax,self.umax)
        f = lambda t,y: [y[1], -self.M*self.g*self.I/self.J*np.sin(y[0]) - 1/self.tau*y[1] + self.Km/self.tau*self.u]
        sol = solve_ivp(f,[0,self.dt],[self.th,self.omega]) #integration
        self.th, self.omega = sol.y[:,-1]
        ##### End do not edit   #####

        reward = self.reward_fun(self)
        return self.get_obs(), reward, False, {}
         
    def reset(self,seed=None):
        self.th = np.random.normal(loc=0,scale=0.001)
        self.omega = np.random.normal(loc=0,scale=0.001)
        self.u = 0
        return self.get_obs()

    def get_obs(self):
        self.th_noise = self.th + np.random.normal(loc=0,scale=0.001) #do not edit
        self.omega_noise = self.omega + np.random.normal(loc=0,scale=0.001) #do not edit
        return [self.th_noise, self.omega_noise]

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.r = 1.25

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.4, 2.4, -2.4, 2.4)

            blue_back = rendering.make_circle(2,res=100)
            blue_back.set_color(24/255,60/255,94/255) #blue
            self.viewer.add_geom(blue_back)

            disk = rendering.make_circle(0.65,res=100)
            disk_mini = rendering.make_circle(0.06,res=30)
            disk.set_color(161/255,143/255,117/255) #grey
            disk_mini.set_color(68/255,59/255,42/255)
            self.disk_transform = rendering.Transform()
            disk.add_attr(self.disk_transform)
            disk_mini.add_attr(self.disk_transform)
            self.viewer.add_geom(disk)
            self.viewer.add_geom(disk_mini)
            # self.disk_transform.set_translation(0,1.2)

            axle = rendering.make_circle(.1)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)

            fname = path.join(path.dirname(__file__), "clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        if self.u:
            self.imgtrans.scale = (+self.u/self.umax, np.abs(self.u/self.umax))
        self.disk_transform.set_rotation(-self.th-np.pi/2)
        self.disk_transform.set_translation(-self.r*np.sin(self.th), -self.r*np.cos(self.th))
        self.viewer.render(return_rgb_array='human' == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class UnbalancedDisk_sincos(UnbalancedDisk):
    """docstring for UnbalancedDisk_sincos"""
    def __init__(self, umax=3., dt = 0.025):
        super(UnbalancedDisk_sincos, self).__init__(umax=umax, dt=dt)
        low = [-1,-1,-40.] 
        high = [1,1,40.]
        self.observation_space = spaces.Box(low=np.array(low,dtype=np.float32),high=np.array(high,dtype=np.float32),shape=(3,))

    def get_obs(self):
        self.th_noise = self.th + np.random.normal(loc=0,scale=0.001) #do not edit
        self.omega_noise = self.omega + np.random.normal(loc=0,scale=0.001) #do not edit
        return np.array([np.sin(self.th_noise), np.cos(self.th_noise), self.omega_noise]) #change anything here

if __name__ == '__main__':
    import time
    env = UnbalancedDisk()

    obs = env.reset()
    env.render()
    try:
        for i in range(100):
            time.sleep(1/24)
            env.step(env.action_space.sample())
            env.render()
    finally:
        env.close()

