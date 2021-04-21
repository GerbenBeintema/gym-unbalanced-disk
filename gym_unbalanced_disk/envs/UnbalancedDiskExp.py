
import gym
from gym import spaces
import numpy as np
from scipy.integrate import solve_ivp
from os import path
import time

global eng, eng_active
eng_active = False

class UnbalancedDisk_exp(gym.Env):
    '''
    UnbalancedDisk_exp
    th =            
                    0
                    |
           np.pi/2-----np.pi*3/2
                    |
                  np.pi = starting location

    '''
    def __init__(self, umax=4., dt=0.025, force_restart_matlab_eng=False):
        global eng, eng_active
        self.connected = False
        if eng_active:
            self.eng = eng
        if not eng_active or force_restart_matlab_eng:
            self.init_matlab()

        self.umax = umax
        self.dt = dt
 
        ### Gym things
        self.action_space = spaces.Box(low=-umax,high=umax,shape=tuple()) #continues
        low = [-float('inf'),-30.]
        high = [float('inf'),30.]
        self.observation_space = spaces.Box(low=np.array(low,dtype=np.float32),high=np.array(high,dtype=np.float32),shape=(2,))


        self.reward_fun = lambda self: np.exp(-((self.th)%(2*np.pi)-np.pi)**2/(2*(np.pi/7)**2)) #example reward function, change this!

        #Viewer things
        self.viewer = None
        self.u = 0 #for visual

        #connect to setup
        self.try_connect()
        if self.connected:
            self.reset()


    def init_matlab(self):
        global eng, eng_active
        try: #try closing the engine
            eng.exit()
        except NameError:
            pass
        p1 = path.join(path.dirname(__file__),'matlabfiles/')
        p2 = path.join(path.dirname(__file__),'matlabfiles/FUGIboardMatlab/')
        print('starting matlab engine...',end='')
        from matlab import engine
        eng = engine.start_matlab()
        print('done')
        self.eng = eng
        eng_active = True
        eng.addpath(p1)
        eng.addpath(p2)
        eng.load_api(nargout=0)


    def try_connect(self):
        global eng
        try:
            print('connecting to experimental setup...',end='')
            eng.fugiboard('CloseAll',nargout=0)                 # Close port
            self.H = eng.fugiboard('Open', 'mops1')             # Open port
            self.H['WatchdogTimeout'] = 5.                      # Watchdog timeout
                                                                # output will go to zero after 5 seconds
            eng.fugiboard('SetParams', self.H)                  # Set the parameter
            eng.fugiboard('Write', self.H, 1., 1., 0., 0.)      # Dummy write to sync interface board
            eng.fugiboard('Write', self.H, 1., 1., 0., 0.)      # Reset position
            eng.fugiboard('Write', self.H, 0., 1., 0., 0.)      # End reset
            for i in range(10):
                tmp = eng.fugiboard('Read',self.H)              # Dummy read sensor data                
            self.connected = True
            print('done')
        except Exception as e:
            self.connected = False
            raise e

    def step(self, action):
        if not self.connected:
            raise ValueError('not connected, use env.try_connect')
        #convert action to u
        self.u = np.clip(action,-self.umax,self.umax)

        #apply action
        global eng
        eng.fugiboard('Write',self.H,0.,1.,float(self.u),0.)

        time.sleep(self.dt)

        obs = self.get_obs()
        reward = self.reward_fun(self)
        return obs, reward, False, {}
        
    def reset(self,seed=None):
        if not self.connected:
            raise ValueError('not connected, use env.try_connect')
        global eng
        eng.fugiboard('Write', self.H, 0., 1., 0., 0.0)
        #wait until readout does not change and 
        omega_now = self.get_obs()[1] #finish this
        t_start = time.time()
        while time.time()-t_start<10: #timeout
            time.sleep(0.1)
            omega_new = self.get_obs()[1]
            if abs(omega_new-omega_now)==0:
                break
            omega_now = omega_new

        time.sleep(0.1)
        eng.fugiboard('Write', self.H, 1., 1., 0., 0.)          # Reset position
        return self.get_obs()

    def get_obs(self):
        if not self.connected:
            raise ValueError('not connected, use env.try_connect')
        global eng
        self.obs_raw = np.array(eng.fugiboard('Read',self.H))[:,0]
        #0 = binary
        #1 = sample time something
        #2 = theta
        #3 = omega
        #4 = force?
        #5 = 4.ss??
        #6 = 3.23
        #7 = 0
        #8 = -inf?

        #obs[2]: 2 theta
        self.th = self.obs_raw[2]
        #obs[3]: 3 omega
        self.omega = self.obs_raw[3]
        return [self.th, self.omega]

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

    def close_viewer(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def close(self):
        global eng, eng_active
        if self.connected:
            eng.exit()
            eng_active = False
            self.connected = False
        self.close_viewer()

class UnbalancedDisk_exp_sincos(UnbalancedDisk_exp):
    """docstring for UnbalancedDisk_exp_sincos"""
    def __init__(self, umax=4, dt = 0.025):
        super(UnbalancedDisk_exp_sincos, self).__init__(umax=umax, dt=dt)
        low = [-1,-1,-40.] 
        high = [1,1,40.]
        self.observation_space = spaces.Box(low=np.array(low,dtype=np.float32),high=np.array(high,dtype=np.float32),shape=(3,))

    def get_obs(self):
        super(UnbalancedDisk_exp_sincos, self).get_obs()
        return np.array([np.sin(self.th), np.cos(self.th), self.omega])

