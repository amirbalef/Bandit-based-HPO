import numpy as np 
from sklearn.preprocessing import MinMaxScaler
   
class QuantileUCB():
    def __init__(self, narms, alpha = 0.25, tau=0.95, noise = False):
        self.narms = narms
        self.alpha = alpha
        self.tau = tau
        self.t = 1
        self.pulled_arms = []
        self.rewards = []
        self.raw_rewards = []
        self.selected_arm = 0
        self.scaler = MinMaxScaler()
        self.intial_steps = 1
        self.noise = noise
        self.random_init = False

    def play(self, context=None):
        pulled_arms = np.bincount(  self.pulled_arms,  minlength=self.narms)
        if np.any(pulled_arms<self.intial_steps):
            if(self.random_init):
                self.selected_arm =  np.random.choice(np.where(pulled_arms<self.intial_steps)[0])
            else:
                self.selected_arm =  (self.t -1)%self.narms
        elif self.noise == True:
            qunatile_rewards = np.asarray([ np.quantile( self.rewards[ np.asarray(self.pulled_arms) == x ] , self.tau)  for x  in range(self.narms)])
            pad =   np.sqrt(self.alpha *(np.log(self.t)) / pulled_arms)
            ucb_values= qunatile_rewards +  pad
            self.selected_arm =  np.argmax(ucb_values)  
        else:
            qunatile_rewards = np.asarray([ np.quantile( self.rewards[ np.asarray(self.pulled_arms) == x ] , self.tau)  for x  in range(self.narms)])
            ucb_values=  qunatile_rewards + np.sqrt(self.alpha *(np.log(self.t)) / pulled_arms)
            self.selected_arm =  np.argmax(ucb_values)  
        return self.selected_arm
    
    
    def update(self, reward, arm = None, context=None):
        if arm ==None:
            arm = self.selected_arm
        self.pulled_arms.append(arm)
        self.rewards.append(reward)
        self.t += 1

    def update_cost(self, cost, arm = None, context=None):
        if arm ==None:
            arm = self.selected_arm
        self.pulled_arms.append(arm)
        self.raw_rewards.append(-cost)
        self.rewards = self.scaler.fit_transform( np.asarray(self.raw_rewards).reshape(-1,1))

        self.t += 1