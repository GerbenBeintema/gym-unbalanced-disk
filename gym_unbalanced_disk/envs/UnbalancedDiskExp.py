
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
from os import path
import time
import usb.util

global dev, dev_active
dev_active = False

#todo:
# update documentation on install and usage

class UnbalancedDisk_exp(gym.Env):
    '''
    UnbalancedDisk_exp
    th =            
                  +-pi
                    |
           pi/2   ----- -pi/2
                    |
                    0  = starting location

    '''
    def __init__(self, umax=3., dt=0.025, force_restart_dev=False, inactivity_release_time=3, render_mode='human'):
        '''
        umax : the maximal allowable input
        dt : the sample time
        force_restart_dev : set to true to reset connection
        inactivity_release_time : If the setup has not recived any inputs for ~inactivity_release_time/20 seconds than the input will be set to zero automaticly
        '''
        global dev, dev_active
        if dev_active:
            self.dev = dev
        if not dev_active or force_restart_dev:
            self.init_dev()

        assert isinstance(inactivity_release_time, int)
        self.set_inactivity_release_time(inactivity_release_time)

        self.umax = umax
        self.dt = dt
 
        ### Gym things
        self.action_space = spaces.Box(low=-umax,high=umax,shape=tuple()) # continuous
        low = [-float('inf'),-30.]
        high = [float('inf'),30.]
        self.observation_space = spaces.Box(low=np.array(low,dtype=np.float32),high=np.array(high,dtype=np.float32),shape=(2,))

        self.reward_fun = lambda self: np.exp(-((self.th)%(2*np.pi)-np.pi)**2/(2*(np.pi/7)**2)) #example reward function, change this!

        #Viewer things
        self.render_mode = render_mode
        self.viewer = None
        self.u = 0 #for visual

    def init_encoder(self):
        data_w=[1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0 ]
        self.dev.write(0x02,data_w,2)
        data_w=[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0 ]
        self.dev.write(0x02,data_w,2)

    def set_inactivity_release_time(self, inactivity_release_time):
        self.inactivity_release_time=inactivity_release_time
        data_w=[1,1,0,0,0,0,0,self.inactivity_release_time,0,0,0,0,0,0,0,0 ]
        self.dev.write(0x02,data_w,2)
        data_w=[0,1,0,0,0,0,0,self.inactivity_release_time,0,0,0,0,0,0,0,0 ]
        self.dev.write(0x02,data_w,2)

    def init_dev(self):
        global dev, dev_active
        try: #try closing the engine
            usb.util.dispose_resources(dev)
        except NameError:
            pass
        dev = usb.core.find(idVendor=0x04b4, idProduct=0x8612)
        dev.set_configuration() #this throws and error if python cannot connect to the disk.
        self.dev = dev
        dev_active = True

    def step(self, action):
        #convert action to u
        self.u = action #continuous
        # self.u = [-3,-1,0,1,3][action] #discrate
        # self.u = [-3,3][action] #discrate

        ##### Do not edit whats below ######
        self.u = np.clip(self.u,-self.umax,self.umax)


        DacMin, DacMax, Relais= -10, 10, 1
        digital_input = int((self.u-DacMin)/(DacMax-DacMin)*65536)
        digital_in_sec = divmod(digital_input,256)

        data_pack=[0,0,digital_in_sec[0],0,0,Relais,digital_in_sec[1],self.inactivity_release_time,0,0,0,0,0,0,0,0]
        self.dev.write(0x02,data_pack,10)
        
        start_t = time.time() #a more accurate waiter than time.sleep
        while time.time() - start_t<self.dt:
            pass
        obs = self.get_obs()
        reward = self.reward_fun(self)
        return obs, reward, False, False, {}
        
    def reset(self,seed=None):
        theta_now = self.get_obs()[0]
        t_start = time.time()
        while time.time()-t_start<30:
            time.sleep(0.1)
            theta_new = self.get_obs()[0]
            if abs(theta_new-theta_now)==0:
                break
            theta_now = theta_new
        time.sleep(0.1)
        self.init_encoder()
        return self.get_obs(), {}

    def get_obs(self):
        couldnotreadcounter = 0
        while True:
            try:
                self.data_pack_read=self.dev.read(0x86,16,1)
                break
            except usb.USBError as e:
                print('USB read error')
                couldnotreadcounter += 1
                time.sleep(0.001)
                if couldnotreadcounter>20:
                    raise e
        # self.data_pack_read=self.dev.read(0x86,16,1) #not sure why this works better than one read WL
        data = self.data_pack_read
        # Write: command, digitalout, [dac1( dac2)]
        # Read:  [status elapsedtime position1 position2 motorcurrent motorvoltage externalvoltage digitalin averagespeed1 averagespeed2]
        # Read:  [status elapsedtime position1 position2 motorcurrent beamvoltage pendulumvoltage digitalin]
        if data[4]<128:
            position=2*np.pi*(data[4]*65536+data[3]*256+data[2])/2000
        else:
            position=2*np.pi*(data[4]*65536+data[3]*256+data[2]-16777216)/2000
        
        #0 = binary
        #1 = sample time something
        #2 = theta
        #3 = omega
        #4 = force?
        #5 = 4.ss??
        #6 = 3.23
        #7 = 0
        #8 = -inf?

        d = data
        omega = d[10]*-3.644127510645671 + d[14]*2.01877019753875 + d[12]*1.6121463023483062 + d[9]*-0.013751126061226403 #not entirely correct?

        #obs[2]: 2 theta
        self.th = position
        #obs[3]: 3 omega
        self.omega = omega#self.obs_raw[3]
        return np.array([self.th, self.omega])

    def render(self):
        import pygame
        from pygame import gfxdraw
        
        screen_width = 500
        screen_height = 500

        th = self.th
        omega = self.omega #x = self.state

        if self.viewer is None:
            pygame.init()
            pygame.display.init()
            self.viewer = pygame.display.set_mode((screen_width, screen_height))

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))
        
        gfxdraw.filled_circle( #central blue disk
            self.surf,
            screen_width//2,
            screen_height//2,
            int(screen_width/2*0.65*1.3),
            (32,60,92),
        )
        gfxdraw.filled_circle( #small midle disk
            self.surf,
            screen_width//2,
            screen_height//2,
            int(screen_width/2*0.06*1.3),
            (132,132,126),
        )
        
        from math import cos, sin
        r = screen_width//2*0.40*1.3
        gfxdraw.filled_circle( #disk
            self.surf,
            int(screen_width//2-sin(th)*r), #is direction correct?
            int(screen_height//2-cos(th)*r),
            int(screen_width/2*0.22*1.3),
            (155,140,108),
        )
        gfxdraw.filled_circle( #small nut
            self.surf,
            int(screen_width//2-sin(th)*r), #is direction correct?
            int(screen_height//2-cos(th)*r),
            int(screen_width/2*0.22/8*1.3),
            (71,63,48),
        )
        
        fname = path.join(path.dirname(__file__), "clockwise.png")
        self.arrow = pygame.image.load(fname)
        if self.u:
            if isinstance(self.u, (np.ndarray,list)):
                if self.u.ndim==1:
                    u = self.u[0]
                elif self.u.ndim==0:
                    u = self.u
                else:
                    raise ValueError(f'u={u} is not the correct shape')
            else:
                u = self.u
            arrow_size = abs(float(u)/self.umax*screen_height)*0.25
            Z = (arrow_size, arrow_size)
            arrow_rot = pygame.transform.scale(self.arrow,Z)
            if self.u<0:
                arrow_rot = pygame.transform.flip(arrow_rot, True, False)
                
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.viewer.blit(self.surf, (0, 0))
        if self.u:
            self.viewer.blit(arrow_rot, (screen_width//2-arrow_size//2, screen_height//2-arrow_size//2))
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()

        return True

    def close_viewer(self):
        if self.viewer is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False
            self.viewer = None

    def close(self):
        global dev, dev_active
        if dev_active:
            usb.util.dispose_resources(self.dev)
            dev_active = False
        self.close_viewer()

class UnbalancedDisk_exp_sincos(UnbalancedDisk_exp):
    """docstring for UnbalancedDisk_exp_sincos"""
    def __init__(self, umax=3., dt = 0.025):
        super(UnbalancedDisk_exp_sincos, self).__init__(umax=umax, dt=dt)
        low = [-1,-1,-40.] 
        high = [1,1,40.]
        self.observation_space = spaces.Box(low=np.array(low,dtype=np.float32),high=np.array(high,dtype=np.float32),shape=(3,))

    def get_obs(self):
        super(UnbalancedDisk_exp_sincos, self).get_obs()
        return np.array([np.sin(self.th), np.cos(self.th), self.omega])

